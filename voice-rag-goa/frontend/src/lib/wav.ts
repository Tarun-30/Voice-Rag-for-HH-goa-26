/**
 * Wrap raw PCM16 mono samples in a minimal 44-byte WAV (RIFF) header so the
 * REST /api/transcribe fallback can upload a self-describing file. Only used
 * when the WebSocket path is unavailable — the live path streams headerless PCM
 * and the server wraps it instead.
 */

export function pcm16ToWavBlob(pcm: Int16Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const numChannels = 1;
  const dataSize = pcm.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true); // file size - 8
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true); // byte rate
  view.setUint16(32, numChannels * bytesPerSample, true); // block align
  view.setUint16(34, 8 * bytesPerSample, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  // Copy samples little-endian.
  let offset = 44;
  for (let i = 0; i < pcm.length; i++, offset += 2) {
    view.setInt16(offset, pcm[i], true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}
