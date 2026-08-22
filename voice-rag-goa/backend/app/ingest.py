"""
Ingestion + multi-strategy chunking + index build.

Pipeline
--------
    HF parquet shard  ->  corpus_cache.jsonl (+ queryset.jsonl)
                      ->  chunk with strategy {semantic|hierarchical|sliding}
                      ->  embed (OnnxEmbedder)
                      ->  write index artefact(s) for {memory|lancedb|both}

Why we read parquet directly instead of `datasets.load_dataset`
---------------------------------------------------------------
The hub's datasets-server viewer API (`/rows`, `/first-rows`) returns HTTP 500
for `ai4bharat/MSMARCO-XI`: the passages column is a nested list-of-list and the
server hits `ArrowNotImplementedError: Nested data conversions not implemented
for chunked array outputs`. So we open the parquet shard over `HfFileSystem` and
stream row batches with `pyarrow`, projecting only the leaf columns we need.

That projection is not a micro-optimisation: `passages.Translated_passages` is
271 MB of the 462 MB shard. Dropping it (when we are not indexing translations)
takes time-to-first-batch from ~30 s to ~11 s. And each shard is a *single* row
group, so `read_row_group(0)` would pull the entire 1.1 GB decompressed column
into RAM before returning a single row - `iter_batches` with an early break is
the only way to read the head of the file cheaply.

Run standalone:
    python -m app.ingest                 # build every strategy from config
    python -m app.ingest --strategy semantic --rows 400 --backend memory
    python -m app.ingest --stats         # just print what is already indexed
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from . import config
from . import textutil
from .embeddings import OnnxEmbedder, get_embedder
from .schemas import ChunkStrategyStats

logger = logging.getLogger(__name__)

INDEX_DIR = config.DATA_DIR / "index"

# Leaf columns we pull from the parquet shard. Dotted paths select struct
# children, so pyarrow never materialises the columns we omit.
_BASE_COLUMNS = [
    "query_id",
    "query_type",
    "query",
    "Eng_Query",
    "Answer",
    "Eng_Answer",
    "passages.English_passages",
    "passages.Translated_passages",
    "passages.is_selected",
]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Passage:
    """One candidate passage from the dataset - the unit we chunk."""

    passage_id: str
    text: str
    query_id: int
    query: str
    title: str
    domain: str
    language: str
    is_gold: bool

    def to_json(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, obj: dict) -> "Passage":
        return cls(**obj)


@dataclass(slots=True)
class QuerySpec:
    """One query with its gold-passage labels - the benchmark ground truth."""

    query_id: int
    query: str
    eng_query: str
    answer: str
    domain: str
    language: str
    gold_passage_ids: list[str]

    def to_json(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, obj: dict) -> "QuerySpec":
        return cls(**obj)


@dataclass(slots=True)
class Chunk:
    """One indexed unit. `text` is embedded/searched; `context_text` is read.

    Mirrors schemas.RetrievedChunk's provenance fields so retrieval can hydrate a
    RetrievedChunk from a stored row with no remapping.
    """

    chunk_id: str
    text: str
    context_text: str
    strategy: str
    passage_id: str
    query_id: int
    title: str
    domain: str
    language: str
    token_count: int
    parent_id: str | None
    is_gold: bool

    def to_row(self) -> dict[str, object]:
        row = dataclasses.asdict(self)
        # LanceDB/pyarrow dislike Python None in a string column; use "".
        if row["parent_id"] is None:
            row["parent_id"] = ""
        return row


# --------------------------------------------------------------------------- #
# Source reading
# --------------------------------------------------------------------------- #


def _open_shard(remote_path: str):
    """Return a pyarrow ParquetFile for a shard path inside the HF repo."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=config.HF_TOKEN)
    full = f"datasets/{config.HF_DATASET_REPO}/{remote_path}"
    logger.info("opening shard %s", full)
    return pq.ParquetFile(fs.open(full, "rb"))


def _iter_rows(
    shard_path: str,
    *,
    limit: int,
    include_translated: bool,
    batch_size: int,
) -> Iterator[dict]:
    """Yield source rows (as dicts) from a shard, stopping after `limit`."""
    columns = list(_BASE_COLUMNS)
    if not include_translated:
        columns.remove("passages.Translated_passages")

    parquet = _open_shard(shard_path)
    yielded = 0
    started = time.perf_counter()
    first_batch = True
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        if first_batch:
            logger.info("time-to-first-batch %.1fs", time.perf_counter() - started)
            first_batch = False
        for row in batch.to_pylist():
            yield row
            yielded += 1
            if yielded >= limit:
                return


def _row_to_passages(
    row: dict,
    *,
    language: str,
    include_translated: bool,
) -> tuple[QuerySpec | None, list[Passage]]:
    """Split one source row into a QuerySpec and its candidate Passages."""
    passages_field = row.get("passages") or {}
    english = passages_field.get("English_passages") or []
    translated = passages_field.get("Translated_passages") or []
    selected = passages_field.get("is_selected") or []

    query_id = int(row.get("query_id") or 0)
    domain = (row.get("query_type") or "").strip() or "general"
    eng_query = textutil.sanitize_text(row.get("Eng_Query") or "", max_chars=512)
    native_query = textutil.sanitize_text(row.get("query") or "", max_chars=512)
    answer = textutil.sanitize_text(
        row.get("Eng_Answer") or row.get("Answer") or "", max_chars=2000
    )

    passages: list[Passage] = []
    gold_ids: list[str] = []
    source_list = translated if include_translated and translated else english
    passage_language = language if (include_translated and translated) else "eng"

    for index, raw in enumerate(source_list):
        text = textutil.sanitize_text(raw or "", max_chars=6000)
        if len(text) < 24:  # drop empty / degenerate passages
            continue
        passage_id = f"{query_id}:{index}"
        is_gold = bool(selected[index]) if index < len(selected) else False
        # MS MARCO passages have no title; use the leading clause as a display
        # label for the citation inspector, and the query_type as the domain tag.
        title = textutil.truncate(text.split(".")[0], 80, suffix="")
        passages.append(
            Passage(
                passage_id=passage_id,
                text=text,
                query_id=query_id,
                query=eng_query or native_query,
                title=title,
                domain=domain,
                language=passage_language,
                is_gold=is_gold,
            )
        )
        if is_gold:
            gold_ids.append(passage_id)

    if not passages:
        return None, []

    query_spec = QuerySpec(
        query_id=query_id,
        query=native_query or eng_query,
        eng_query=eng_query or native_query,
        answer=answer,
        domain=domain,
        language=language,
        gold_passage_ids=gold_ids,
    )
    return query_spec, passages


# --------------------------------------------------------------------------- #
# Corpus cache
# --------------------------------------------------------------------------- #


def build_corpus(
    *,
    languages: Sequence[str] | None = None,
    split: str | None = None,
    limit: int | None = None,
    include_translated: bool | None = None,
    force: bool = False,
) -> tuple[list[Passage], list[QuerySpec]]:
    """Build (or load) the passage corpus and query set, cached as JSONL."""
    split = split or config.INGEST_SPLIT
    limit = limit or config.INGEST_ROW_LIMIT
    include_translated = (
        config.INGEST_INCLUDE_TRANSLATED if include_translated is None else include_translated
    )
    shards = (
        config.resolve_languages(list(languages))
        if languages
        else config.resolve_languages()
    )

    corpus_path = Path(config.CORPUS_CACHE_PATH)
    query_path = Path(config.QUERYSET_CACHE_PATH)
    if not force and corpus_path.exists() and query_path.exists():
        passages = _load_jsonl(corpus_path, Passage.from_json)
        queries = _load_jsonl(query_path, QuerySpec.from_json)
        if passages and queries:
            logger.info(
                "corpus cache hit: %d passages, %d queries", len(passages), len(queries)
            )
            return passages, queries

    config.ensure_dirs()
    all_passages: list[Passage] = []
    all_queries: list[QuerySpec] = []
    seen_passage_ids: set[str] = set()

    for shard in shards:
        remote = shard.shard_for(split)
        if remote is None:
            logger.warning("language %s has no %s shard; skipping", shard.key, split)
            continue
        rows = _iter_rows(
            remote,
            limit=limit,
            include_translated=include_translated,
            batch_size=config.INGEST_BATCH_SIZE,
        )
        kept = 0
        for row in rows:
            query_spec, passages = _row_to_passages(
                row, language=shard.key, include_translated=include_translated
            )
            if query_spec is None:
                continue
            # Namespace ids by language so multiple shards never collide.
            prefixed: list[Passage] = []
            for passage in passages:
                passage.passage_id = f"{shard.key}:{passage.passage_id}"
                if passage.passage_id in seen_passage_ids:
                    continue
                seen_passage_ids.add(passage.passage_id)
                prefixed.append(passage)
            if not prefixed:
                continue
            query_spec.query_id = query_spec.query_id
            query_spec.gold_passage_ids = [
                f"{shard.key}:{pid}" for pid in query_spec.gold_passage_ids
            ]
            all_passages.extend(prefixed)
            all_queries.append(query_spec)
            kept += 1
        logger.info("shard %s: kept %d queries", shard.key, kept)

    _write_jsonl(corpus_path, (p.to_json() for p in all_passages))
    _write_jsonl(query_path, (q.to_json() for q in all_queries))
    logger.info(
        "corpus built: %d passages, %d queries -> %s",
        len(all_passages),
        len(all_queries),
        corpus_path.name,
    )
    return all_passages, all_queries


# --------------------------------------------------------------------------- #
# Chunking strategies
# --------------------------------------------------------------------------- #


def _slice_by_tokens(
    text: str,
    spans: list[tuple[int, int]],
    start_tok: int,
    end_tok: int,
) -> str:
    """Return the substring covering token indices [start_tok, end_tok)."""
    if not spans or start_tok >= len(spans):
        return ""
    end_tok = min(end_tok, len(spans))
    start_char = spans[start_tok][0]
    end_char = spans[end_tok - 1][1]
    return text[start_char:end_char].strip()


def chunk_semantic(
    passage: Passage,
    embedder: OnnxEmbedder,
    *,
    sentence_vectors: np.ndarray | None = None,
    sentences: list[str] | None = None,
) -> list[Chunk]:
    """Strategy A - break where adjacent-sentence meaning shifts.

    Embed each sentence, measure cosine *distance* between consecutive sentences
    (with a small look-back buffer to smooth single-sentence noise), and start a
    new chunk wherever that distance spikes above mean + sigma*std. Then merge to
    respect min tokens and hard-split anything over max tokens. This puts chunk
    boundaries at genuine topic shifts instead of arbitrary length cuts.

    `sentence_vectors`/`sentences` may be supplied by the batch builder so all
    sentences across the whole corpus are embedded in one call.
    """
    params = config.CHUNKING
    if sentences is None:
        sentences = textutil.split_sentences(passage.text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return _finalize_semantic_groups(passage, [sentences], embedder)

    if sentence_vectors is None:
        sentence_vectors = embedder.embed_passages(sentences, sort_by_length=False)

    buffer = max(1, params.semantic_buffer_size)
    distances: list[float] = []
    for i in range(len(sentences) - 1):
        left = sentence_vectors[max(0, i - buffer + 1) : i + 1].mean(axis=0)
        right = sentence_vectors[i + 1 : i + 1 + buffer].mean(axis=0)
        left = left / max(float(np.linalg.norm(left)), 1e-12)
        right = right / max(float(np.linalg.norm(right)), 1e-12)
        distances.append(1.0 - float(np.dot(left, right)))

    if distances:
        mean = statistics.fmean(distances)
        std = statistics.pstdev(distances) if len(distances) > 1 else 0.0
        threshold = mean + params.semantic_breakpoint_sigma * std
    else:
        threshold = 1.0

    groups: list[list[str]] = []
    current = [sentences[0]]
    for i, distance in enumerate(distances):
        if distance >= threshold and distance > 1e-6:
            groups.append(current)
            current = []
        current.append(sentences[i + 1])
    if current:
        groups.append(current)

    return _finalize_semantic_groups(passage, groups, embedder)


def _finalize_semantic_groups(
    passage: Passage, groups: list[list[str]], embedder: OnnxEmbedder
) -> list[Chunk]:
    """Enforce min/max token limits on raw semantic groups, then materialise."""
    params = config.CHUNKING
    texts = [" ".join(g).strip() for g in groups if g]
    counts = embedder.count_tokens_batch(texts)

    # Merge forward until each chunk clears the minimum size.
    merged: list[str] = []
    merged_counts: list[int] = []
    for text, count in zip(texts, counts):
        if merged and merged_counts[-1] < params.semantic_min_tokens:
            merged[-1] = f"{merged[-1]} {text}".strip()
            merged_counts[-1] += count
        else:
            merged.append(text)
            merged_counts.append(count)

    # Hard-split anything still over the maximum, on token boundaries.
    chunks: list[Chunk] = []
    index = 0
    for text, count in zip(merged, merged_counts):
        if count <= params.semantic_max_tokens:
            chunks.append(_make_chunk(passage, config.STRATEGY_SEMANTIC, index, text, count))
            index += 1
            continue
        spans = embedder.token_spans(text)
        step = params.semantic_max_tokens
        for start in range(0, len(spans), step):
            piece = _slice_by_tokens(text, spans, start, start + step)
            if piece:
                chunks.append(
                    _make_chunk(
                        passage,
                        config.STRATEGY_SEMANTIC,
                        index,
                        piece,
                        min(step, len(spans) - start),
                    )
                )
                index += 1
    return chunks


def chunk_hierarchical(passage: Passage, embedder: OnnxEmbedder) -> list[Chunk]:
    """Strategy B - small children for matching, large parents for reading.

    Split into parent windows of ~parent_tokens, then each parent into
    child_tokens children with a small overlap. The child text is what gets
    embedded and matched; `context_text` on every child is its *parent*, so a
    precise 128-token match still hands the LLM the full 512-token neighbourhood.
    """
    params = config.CHUNKING
    spans = embedder.token_spans(passage.text)
    if not spans:
        return []

    chunks: list[Chunk] = []
    child_index = 0
    parent_step = params.parent_tokens
    for parent_no, parent_start in enumerate(range(0, len(spans), parent_step)):
        parent_text = _slice_by_tokens(
            passage.text, spans, parent_start, parent_start + parent_step
        )
        if not parent_text:
            continue
        parent_id = f"{passage.passage_id}#p{parent_no}"
        parent_tok_count = min(parent_step, len(spans) - parent_start)

        # Children walk this parent's token span with overlap.
        child_step = max(1, params.child_tokens - params.child_overlap_tokens)
        parent_end = min(parent_start + parent_step, len(spans))
        local = parent_start
        while local < parent_end:
            child_text = _slice_by_tokens(
                passage.text, spans, local, min(local + params.child_tokens, parent_end)
            )
            if child_text:
                chunk = _make_chunk(
                    passage,
                    config.STRATEGY_HIERARCHICAL,
                    child_index,
                    child_text,
                    min(params.child_tokens, parent_end - local),
                    parent_id=parent_id,
                )
                # The whole point of Strategy B: read the parent, match the child.
                chunk.context_text = parent_text
                chunks.append(chunk)
                child_index += 1
            local += child_step
        # Guard against a parent that produced no child (all-whitespace tail).
        if not any(c.parent_id == parent_id for c in chunks):
            chunk = _make_chunk(
                passage,
                config.STRATEGY_HIERARCHICAL,
                child_index,
                parent_text,
                parent_tok_count,
                parent_id=parent_id,
            )
            chunks.append(chunk)
            child_index += 1
    return chunks


def chunk_sliding(passage: Passage, embedder: OnnxEmbedder) -> list[Chunk]:
    """Strategy C - fixed windows with overlap, metadata folded into the vector.

    A metadata tag (title / passage id / domain) is prepended to the searchable
    text, so the embedding and the BM25 postings both carry provenance. Windows
    are window_tokens wide with a 25% overlap so an answer straddling a boundary
    still lives whole inside some window.
    """
    params = config.CHUNKING
    spans = embedder.token_spans(passage.text)
    if not spans:
        return []

    overlap = int(params.window_tokens * params.window_overlap_ratio)
    step = max(1, params.window_tokens - overlap)
    tag = f"[title: {passage.title} | id: {passage.passage_id} | domain: {passage.domain}] "

    chunks: list[Chunk] = []
    index = 0
    for start in range(0, len(spans), step):
        body = _slice_by_tokens(passage.text, spans, start, start + params.window_tokens)
        if not body:
            continue
        searchable = tag + body
        chunk = _make_chunk(
            passage,
            config.STRATEGY_SLIDING,
            index,
            searchable,
            min(params.window_tokens, len(spans) - start),
        )
        # Embed/search the tagged text, but read the clean body.
        chunk.context_text = body
        chunks.append(chunk)
        index += 1
        if start + params.window_tokens >= len(spans):
            break
    return chunks


def _make_chunk(
    passage: Passage,
    strategy: str,
    index: int,
    text: str,
    token_count: int,
    *,
    parent_id: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"{passage.passage_id}#{strategy[:3]}{index}",
        text=text,
        context_text=text,
        strategy=strategy,
        passage_id=passage.passage_id,
        query_id=passage.query_id,
        title=passage.title,
        domain=passage.domain,
        language=passage.language,
        token_count=token_count,
        parent_id=parent_id,
        is_gold=passage.is_gold,
    )


def chunk_passages(
    passages: Sequence[Passage], strategy: str, embedder: OnnxEmbedder
) -> list[Chunk]:
    """Chunk a whole corpus under one strategy.

    For semantic, every sentence in the corpus is embedded in a single batched
    call and sliced back per passage - one big matmul instead of thousands of
    tiny ones.
    """
    if strategy == config.STRATEGY_SEMANTIC:
        per_passage_sentences = [textutil.split_sentences(p.text) for p in passages]
        flat: list[str] = []
        offsets: list[tuple[int, int]] = []
        for sentences in per_passage_sentences:
            start = len(flat)
            flat.extend(sentences)
            offsets.append((start, len(flat)))
        logger.info("semantic: embedding %d sentences in one batch", len(flat))
        vectors = (
            embedder.embed_passages(flat, batch_size=64, sort_by_length=True)
            if flat
            else np.zeros((0, embedder.dim), dtype=np.float32)
        )
        chunks: list[Chunk] = []
        for passage, sentences, (lo, hi) in zip(passages, per_passage_sentences, offsets):
            chunks.extend(
                chunk_semantic(
                    passage,
                    embedder,
                    sentences=sentences,
                    sentence_vectors=vectors[lo:hi] if hi > lo else None,
                )
            )
        return chunks

    chunker = {
        config.STRATEGY_HIERARCHICAL: chunk_hierarchical,
        config.STRATEGY_SLIDING: chunk_sliding,
    }[strategy]
    chunks = []
    for passage in passages:
        chunks.extend(chunker(passage, embedder))
    return chunks


# --------------------------------------------------------------------------- #
# Index build
# --------------------------------------------------------------------------- #


def build_index(
    strategy: str,
    *,
    backend: str | None = None,
    passages: Sequence[Passage] | None = None,
    embedder: OnnxEmbedder | None = None,
    force: bool = False,
) -> ChunkStrategyStats:
    """Chunk, embed, and persist one strategy's index for the chosen backend(s)."""
    if strategy not in config.ALL_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    backend = backend or config.RETRIEVAL_BACKEND
    embedder = embedder or get_embedder()
    if passages is None:
        passages, _ = build_corpus()

    started = time.perf_counter()
    logger.info("chunking %d passages with strategy=%s", len(passages), strategy)
    chunks = chunk_passages(passages, strategy, embedder)
    if not chunks:
        raise RuntimeError(f"strategy {strategy} produced no chunks")

    logger.info("embedding %d chunks", len(chunks))
    vectors = embedder.embed_passages(
        [c.text for c in chunks],
        batch_size=64,
        sort_by_length=True,
        on_progress=_progress("embed"),
    )
    build_seconds = time.perf_counter() - started

    if backend in ("memory", "both"):
        _write_memory_index(strategy, chunks, vectors, embedder, build_seconds)
    if backend in ("lancedb", "both"):
        _write_lancedb_index(strategy, chunks, vectors, embedder)

    stats = _compute_stats(strategy, chunks, embedder, build_seconds, backend)
    logger.info(
        "index %s done: %d chunks, %.1fs, avg %.0f tokens",
        strategy,
        stats.rows,
        build_seconds,
        stats.avg_tokens,
    )
    return stats


def _write_memory_index(
    strategy: str,
    chunks: list[Chunk],
    vectors: np.ndarray,
    embedder: OnnxEmbedder,
    build_seconds: float,
) -> None:
    """Persist float32 matrix (.npy) + chunk metadata (.parquet) + manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = INDEX_DIR / strategy
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vectors.astype(np.float32))

    rows = [c.to_row() for c in chunks]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out / "chunks.parquet")

    manifest = {
        "strategy": strategy,
        "rows": len(chunks),
        "dim": int(vectors.shape[1]) if vectors.size else embedder.dim,
        "embed_model": embedder.model_name,
        "build_seconds": round(build_seconds, 3),
        "has_parents": any(c.parent_id for c in chunks),
        "backend": "memory",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_lancedb_index(
    strategy: str,
    chunks: list[Chunk],
    vectors: np.ndarray,
    embedder: OnnxEmbedder,
) -> None:
    """Persist a LanceDB table with an ANN index and a native BM25 FTS index."""
    import lancedb

    config.ensure_dirs()
    db = lancedb.connect(str(config.LANCEDB_URI))
    name = config.table_name(strategy)
    rows = []
    for chunk, vector in zip(chunks, vectors):
        row = chunk.to_row()
        row["vector"] = vector.astype(np.float32)
        rows.append(row)

    try:
        db.drop_table(name)
    except Exception:
        pass
    table = db.create_table(name, data=rows)

    # Full-text index on the searchable text (tantivy backend).
    try:
        table.create_fts_index("text", replace=True, use_tantivy=False)
    except Exception as exc:  # pragma: no cover - FTS is optional at small sizes
        logger.warning("FTS index on %s failed: %s", name, exc)

    # ANN index only pays off past a few thousand rows; below that the brute
    # force scan is faster and exact.
    if len(chunks) >= config.ANN_MIN_ROWS:
        try:
            partitions = max(1, len(chunks) // config.IVF_PARTITION_DIVISOR)
            table.create_index(
                metric=config.VECTOR_METRIC,
                num_partitions=partitions,
                num_sub_vectors=min(96, embedder.dim // 4),
                vector_column_name="vector",
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("ANN index on %s failed: %s", name, exc)


def _compute_stats(
    strategy: str,
    chunks: list[Chunk],
    embedder: OnnxEmbedder,
    build_seconds: float,
    backend: str,
) -> ChunkStrategyStats:
    counts = [c.token_count for c in chunks]
    counts_sorted = sorted(counts)
    descriptions = {
        config.STRATEGY_SEMANTIC: "Sentence-embedding breakpoint detection: "
        "chunks split where adjacent-sentence cosine distance spikes.",
        config.STRATEGY_HIERARCHICAL: "Parent/child: 128-token children matched, "
        "512-token parents read by the LLM.",
        config.STRATEGY_SLIDING: "256-token windows, 25% overlap, "
        "title/id/domain folded into the embedded text.",
    }
    return ChunkStrategyStats(
        strategy=strategy,  # type: ignore[arg-type]
        table=config.table_name(strategy),
        rows=len(chunks),
        unique_passages=len({c.passage_id for c in chunks}),
        avg_tokens=round(statistics.fmean(counts), 1) if counts else 0.0,
        min_tokens=counts_sorted[0] if counts else 0,
        max_tokens=counts_sorted[-1] if counts else 0,
        p50_tokens=float(counts_sorted[len(counts_sorted) // 2]) if counts else 0.0,
        has_parents=any(c.parent_id for c in chunks),
        has_ann_index=backend in ("lancedb", "both") and len(chunks) >= config.ANN_MIN_ROWS,
        has_fts_index=backend in ("lancedb", "both"),
        build_seconds=round(build_seconds, 3),
        description=descriptions.get(strategy, ""),
    )


def build_all(
    *,
    strategies: Sequence[str] | None = None,
    backend: str | None = None,
    force: bool = False,
) -> list[ChunkStrategyStats]:
    """Build every requested strategy from a single shared corpus + embedder."""
    strategies = strategies or config.ALL_STRATEGIES
    embedder = get_embedder()
    embedder.warmup()
    passages, _ = build_corpus(force=force)
    stats: list[ChunkStrategyStats] = []
    for strategy in strategies:
        if not force and _index_exists(strategy, backend or config.RETRIEVAL_BACKEND):
            logger.info("strategy %s already indexed; skipping (use force=True)", strategy)
            stats.append(load_stats(strategy) or _placeholder_stats(strategy))
            continue
        stats.append(build_index(strategy, backend=backend, passages=passages, embedder=embedder))
    return stats


# --------------------------------------------------------------------------- #
# Introspection helpers used by retrieval / main / benchmark
# --------------------------------------------------------------------------- #


def _index_exists(strategy: str, backend: str) -> bool:
    if backend in ("memory", "both"):
        out = INDEX_DIR / strategy
        if (out / "vectors.npy").exists() and (out / "chunks.parquet").exists():
            return True
    if backend == "lancedb":
        try:
            import lancedb

            db = lancedb.connect(str(config.LANCEDB_URI))
            return config.table_name(strategy) in db.table_names()
        except Exception:
            return False
    return False


def load_stats(strategy: str) -> ChunkStrategyStats | None:
    manifest = INDEX_DIR / strategy / "manifest.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    # Recompute token stats lazily from the parquet if the manifest predates them.
    return _placeholder_stats(strategy, rows=data.get("rows", 0), build_seconds=data.get("build_seconds", 0.0))


def _placeholder_stats(strategy: str, *, rows: int = 0, build_seconds: float = 0.0) -> ChunkStrategyStats:
    return ChunkStrategyStats(
        strategy=strategy,  # type: ignore[arg-type]
        table=config.table_name(strategy),
        rows=rows,
        build_seconds=build_seconds,
    )


def indexed_strategies(backend: str | None = None) -> list[str]:
    backend = backend or config.RETRIEVAL_BACKEND
    return [s for s in config.ALL_STRATEGIES if _index_exists(s, backend)]


# --------------------------------------------------------------------------- #
# JSONL io
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path, factory) -> list:
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(factory(json.loads(line)))
    return out


def load_queryset() -> list[QuerySpec]:
    path = Path(config.QUERYSET_CACHE_PATH)
    if not path.exists():
        build_corpus()
    return _load_jsonl(path, QuerySpec.from_json)


def _progress(label: str):
    last = {"pct": -10}

    def report(done: int, total: int) -> None:
        pct = int(done * 100 / max(1, total))
        if pct >= last["pct"] + 20 or done == total:
            last["pct"] = pct
            logger.info("  %s %d/%d (%d%%)", label, done, total, pct)

    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MSMARCO-XI indexes.")
    parser.add_argument("--strategy", choices=list(config.ALL_STRATEGIES))
    parser.add_argument("--backend", choices=["memory", "lancedb", "both"])
    parser.add_argument("--rows", type=int, help="override INGEST_ROW_LIMIT")
    parser.add_argument("--languages", help="comma-separated language keys")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    parser.add_argument("--stats", action="store_true", help="print index stats and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.stats:
        for strategy in config.ALL_STRATEGIES:
            exists = _index_exists(strategy, config.RETRIEVAL_BACKEND)
            print(f"  {strategy:14s} indexed={exists}")
        return

    if args.rows:
        config.INGEST_ROW_LIMIT = args.rows  # type: ignore[misc]
    languages = args.languages.split(",") if args.languages else None
    if languages:
        passages, _ = build_corpus(languages=languages, force=args.force)
    else:
        passages = None

    strategies = [args.strategy] if args.strategy else config.ALL_STRATEGIES
    embedder = get_embedder()
    embedder.warmup()
    if passages is None:
        passages, _ = build_corpus(force=args.force)
    for strategy in strategies:
        build_index(strategy, backend=args.backend, passages=passages, embedder=embedder, force=args.force)


if __name__ == "__main__":
    main()
