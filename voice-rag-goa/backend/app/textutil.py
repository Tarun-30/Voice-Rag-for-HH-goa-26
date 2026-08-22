"""
Shared text utilities: sentence segmentation, BM25 tokenisation, sanitisation.

Lives in its own module because three very different consumers need the same
behaviour and must agree on it:

* `ingest.py`   - the semantic chunker breaks on sentence boundaries.
* `retrieval.py`- BM25 needs the same tokenisation at index and query time, or
                  the term statistics silently stop lining up.
* `guardrails.py` - the grounding scorer verifies claims sentence by sentence.

No heavy imports here (no pyarrow, no lancedb, no onnxruntime) so the guardrail
path stays cheap to load.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final, Iterable

# --------------------------------------------------------------------------- #
# Sentence segmentation
# --------------------------------------------------------------------------- #

# Terminators across the scripts this dataset actually contains: Latin full
# stop/!/?, Devanagari danda + double danda (Hindi/Marathi), Arabic full stop
# (Urdu), and CJK/fullwidth forms for completeness.
_TERMINATORS: Final[str] = ".!?।॥۔。！？⁉‽"
_CLOSERS: Final[str] = "\"'”’)]}»"

_BOUNDARY = re.compile(
    rf"([{re.escape(_TERMINATORS)}]+)([{re.escape(_CLOSERS)}]*)(\s+|$)"
)

# Words that end in a period without ending a sentence. Deliberately kept to
# high-frequency cases seen in MS MARCO web text - a bigger list costs recall of
# real boundaries for very little gain.
_ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "rev", "hon", "sr", "jr", "st", "mt",
        "vs", "etc", "inc", "ltd", "llc", "co", "corp", "dept", "div", "est",
        "fig", "figs", "no", "nos", "vol", "vols", "pp", "ed", "eds", "al",
        "approx", "avg", "min", "max", "ca", "cf", "eg", "ie", "viz",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
        "nov", "dec", "mon", "tue", "tues", "wed", "thu", "thur", "thurs",
        "fri", "sat", "sun",
        "u.s", "u.k", "u.n", "e.g", "i.e", "a.m", "p.m", "d.c", "ph.d", "b.c",
        "a.d", "sq", "ft", "lb", "lbs", "oz", "kg", "km", "cm", "mm", "hr",
        "hrs", "sec", "yr", "yrs", "tbsp", "tsp", "govt", "assn", "univ",
    }
)

_WORD_BEFORE_DOT = re.compile(r"([A-Za-z\.]+)$")


def _is_false_boundary(text: str, dot_index: int) -> bool:
    """True when the terminator at `dot_index` does not end a sentence."""
    if text[dot_index] != ".":
        # ! ? and the Indic dandas are unambiguous terminators.
        return False

    head = text[:dot_index]
    match = _WORD_BEFORE_DOT.search(head)
    if match:
        word = match.group(1).lower().strip(".")
        # "Dr." / "e.g." / "U.S." - abbreviation, not a boundary.
        if word in _ABBREVIATIONS:
            return True
        # "J. R. R. Tolkien" - a single-letter initial.
        if len(word) == 1 and word.isalpha():
            return True
        # Any dotted initialism - "U.S.C.", "N.A.S.A.", "Ph.D." - generalises
        # past the hardcoded list: every alphabetic run between dots is a
        # single letter, or the whole thing is a known abbreviation stem.
        if "." in word:
            runs = [r for r in word.split(".") if r]
            if runs and all(len(r) == 1 for r in runs):
                return True
            if runs and runs[-1] in _ABBREVIATIONS:
                return True

    # "3.5 million" / "1. First item" - a digit on both sides, or a digit
    # immediately before with a digit after.
    before = head[-1:] if head else ""
    after = text[dot_index + 1 : dot_index + 2]
    if before.isdigit() and after.isdigit():
        return True

    # An ellipsis mid-sentence ("wait ... and then") is not a boundary; a
    # terminal one is handled because the next char will be whitespace + capital.
    if text[dot_index : dot_index + 3] == "..." and after not in ("", " "):
        return True

    return False


def split_sentences(text: str, *, min_chars: int = 2) -> list[str]:
    """Split `text` into sentences.

    A hand-rolled splitter rather than nltk/spacy/pysbd: those add a 40-500 MB
    dependency and a model download to a service whose whole selling point is
    cold-start latency, and none of them handle the Devanagari danda out of the
    box anyway.

    Fragments shorter than `min_chars` are merged into the previous sentence so a
    stray "A." never becomes its own chunk.
    """
    text = (text or "").strip()
    if not text:
        return []

    pieces: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        terminator_start = match.start(1)
        # Check the last character of the terminator run (handles "?!" and "...").
        if _is_false_boundary(text, match.end(1) - 1):
            continue
        end = match.end(2)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = match.end(3)

    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    if not pieces:
        return [text]

    # Merge runts forward-into-previous.
    merged: list[str] = []
    for piece in pieces:
        if merged and len(piece) < min_chars:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, dropping empties."""
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

_WS = re.compile(r"[ \t  -   　]+")
_NEWLINES = re.compile(r"\n{3,}")
# C0/C1 controls except \t \n \r, plus zero-width and bidi-override characters.
# The bidi ones matter: they are a real prompt-injection vector (visually hidden
# instructions), so sanitisation strips rather than preserves them.
_CONTROL = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏‪-‮⁦-⁩﻿]"
)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs, cap blank-line runs, strip the ends."""
    if not text:
        return ""
    text = _WS.sub(" ", text)
    text = _NEWLINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def strip_control_chars(text: str) -> str:
    """Remove invisible characters used to smuggle instructions past a reader."""
    return _CONTROL.sub("", text or "")


def sanitize_text(text: str, *, max_chars: int | None = None) -> str:
    """Full cleanup pass: NFKC fold, de-invisible, collapse whitespace, clamp.

    NFKC matters for guardrails: it folds fullwidth "ｉｇｎｏｒｅ" and other
    homoglyph-ish forms back to ASCII so pattern matching cannot be dodged by
    using a different Unicode presentation of the same word.
    """
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = strip_control_chars(cleaned)
    cleaned = normalize_whitespace(cleaned)
    if max_chars is not None and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
    return cleaned


# --------------------------------------------------------------------------- #
# BM25 tokenisation
# --------------------------------------------------------------------------- #

# Python's `\w` covers letters, digits and underscore - but NOT Unicode
# combining marks (categories Mn/Mc/Me). That is fatal for Indic text: every
# Devanagari matra is a mark, so `\w+` on "भारत की राजधानी" returns
# ['भ','रत','क','र','जध','न','क','य','ह'] - the words are shredded into
# fragments and BM25 becomes noise. Since this dataset is *specifically*
# 14 Indian languages, the tokeniser has to know about marks.
#
# So build the word pattern by asking Unicode itself which codepoints are marks,
# over the range that contains every combining mark for the scripts in play
# (U+0300 Latin diacritics through U+1AFF, plus the combining-mark blocks at
# U+20D0 and U+FE20). Costs ~2 ms once at import; correct for every script
# rather than just the ones someone remembered to hardcode.
def _build_word_pattern() -> re.Pattern[str]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    previous = -2
    for codepoint in range(0x0300, 0x1B00):
        if unicodedata.category(chr(codepoint)) in ("Mn", "Mc", "Me"):
            if start is None:
                start = codepoint
            previous = codepoint
        elif start is not None:
            ranges.append((start, previous))
            start = None
    if start is not None:
        ranges.append((start, previous))
    ranges += [(0x20D0, 0x20F0), (0xFE20, 0xFE2F)]

    marks = "".join(
        chr(lo) if lo == hi else f"{chr(lo)}-{chr(hi)}" for lo, hi in ranges
    )
    # A token is a run of word characters and/or combining marks. `_` is excluded
    # via [^\W_] so "foo_bar" splits into two terms, matching search intuition.
    return re.compile(rf"(?:[^\W_]|[{marks}])+", re.UNICODE)


_TOKEN = _build_word_pattern()

# A small English stoplist. Removing these from BM25 postings is a measurable
# precision win on MS MARCO, where nearly every query starts with "what is the".
STOPWORDS: Final[frozenset[str]] = frozenset(
    """
    a an and are as at be been being but by for from had has have he her hers him
    his how i if in into is it its of on or our ours she that the their theirs
    them then there these they this to was were what when where which while who
    whom why will with you your yours do does did doing done can could should
    would may might must shall about above after again against all also am any
    because before below between both during each few further here more most no
    nor not now only other out over own same so some such than too under until
    up very via
    """.split()
)


def tokenize_for_bm25(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase word tokens for sparse scoring.

    Must be used identically at index time and query time. Numbers are kept -
    MS MARCO is full of "how many", "what year", "$", and dropping digits
    destroys those queries.
    """
    if not text:
        return []
    tokens = _TOKEN.findall(unicodedata.normalize("NFKC", text).lower())
    if drop_stopwords:
        kept = [t for t in tokens if t not in STOPWORDS]
        # Never return empty for a non-empty input: an all-stopword query such
        # as "what is it" would otherwise score zero against everything.
        return kept or tokens
    return tokens


def keyword_overlap(a: str, b: str) -> float:
    """Jaccard overlap of content words. Cheap lexical off-topic pre-filter."""
    left = set(tokenize_for_bm25(a))
    right = set(tokenize_for_bm25(b))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def coverage(needle: str, haystack: str) -> float:
    """Fraction of `needle`'s content words that appear in `haystack`.

    Asymmetric on purpose - used to ask "is this claim's vocabulary present in
    the retrieved context", which is a directional question.
    """
    words = tokenize_for_bm25(needle)
    if not words:
        return 0.0
    pool = set(tokenize_for_bm25(haystack))
    return sum(1 for w in words if w in pool) / len(words)


def looks_like_gibberish(text: str) -> tuple[bool, str]:
    """Heuristic junk detector for the input guardrail.

    Catches the realistic failure mode: an ASR hallucination on silence, or a
    keyboard mash. Returns (is_gibberish, reason). Deliberately conservative -
    a false positive here refuses a legitimate user, so every rule needs a wide
    margin, and any text containing several real words is let through.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True, "empty"

    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        # Pure digits/punctuation. "2024?" is a plausible query, "!!!!" is not.
        if any(c.isdigit() for c in stripped):
            return False, ""
        return True, "no alphanumeric content"

    # Symbol soup: more than half non-alphanumeric, non-space.
    symbols = sum(1 for c in stripped if not (c.isalnum() or c.isspace()))
    if len(stripped) >= 8 and symbols / len(stripped) > 0.5:
        return True, "majority punctuation"

    words = _TOKEN.findall(stripped.lower())
    if not words:
        return True, "no word tokens"

    # A single very long unbroken "word" is a keyboard mash, not a question.
    if len(words) == 1 and len(words[0]) > 24:
        return True, "single overlong token"

    # Vowel-free alphabetic words are the classic mash signature. Only fire when
    # essentially the whole input looks that way, and only for Latin script -
    # abugidas legitimately write without standalone vowel characters.
    latin_words = [w for w in words if w.isascii() and w.isalpha() and len(w) >= 4]
    if latin_words and len(latin_words) == len(words):
        vowelless = [w for w in latin_words if not set(w) & set("aeiouy")]
        if len(vowelless) == len(latin_words):
            return True, "no vowels in any word"

    return False, ""


def truncate(text: str, limit: int, suffix: str = "...") -> str:
    """Character-truncate on a word boundary where possible."""
    text = text or ""
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - len(suffix))]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip() + suffix


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
