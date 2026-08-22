/**
 * Microphone capture -> PCM16 @ 16 kHz, the format both Sarvam and Groq Whisper
 * expect once WAV-wrapped server-side.
 *
 * The browser's AudioContext runs at 44.1/48 kHz; we capture the full mono
 * Float32 stream and resample once at stop() with linear interpolation, which
 * avoids the phase artifacts of resampling each 128-sample block independently.
 * An AnalyserNode is exposed so the UI can draw a live waveform from the same
 * stream without a second getUserMedia.
 *
 * Capture prefers an AudioWorklet (off the main thread) and falls back to the
 * deprecated-but-universal ScriptProcessorNode when AudioWorklet is missing.
 */

const WORKLET_SOURCE = `
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._count = 0;
  }
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length) {
      this._buffer.push(channel.slice(0));
      this._count += channel.length;
      if (this._count >= 2048) {
        const merged = new Float32Array(this._count);
        let offset = 0;
        for (const block of this._buffer) { merged.set(block, offset); offset += block.length; }
        this.port.postMessage(merged, [merged.buffer]);
        this._buffer = [];
        this._count = 0;
      }
    }
    return true;
  }
}
registerProcessor('capture-processor', CaptureProcessor);
`;

function floatToPcm16(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function resample(input: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate || input.length === 0) return input;
  const ratio = fromRate / toRate;
  const newLength = Math.floor(input.length / ratio);
  const out = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const position = i * ratio;
    const index = Math.floor(position);
    const frac = position - index;
    const a = input[index] ?? 0;
    const b = index + 1 < input.length ? input[index + 1] : a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

export interface MicRecorderOptions {
  targetSampleRate?: number;
}

export class MicRecorder {
  readonly targetSampleRate: number;

  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private worklet: AudioWorkletNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private sink: GainNode | null = null;
  private _analyser: AnalyserNode | null = null;
  private blocks: Float32Array[] = [];
  private _recording = false;

  constructor(options: MicRecorderOptions = {}) {
    this.targetSampleRate = options.targetSampleRate ?? 16000;
  }

  get recording(): boolean {
    return this._recording;
  }

  get analyser(): AnalyserNode | null {
    return this._analyser;
  }

  get inputSampleRate(): number {
    return this.context?.sampleRate ?? 48000;
  }

  /** Request the mic and begin capturing. Throws if permission is denied. */
  async start(): Promise<void> {
    if (this._recording) return;
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("Microphone capture is not supported in this browser.");
    }

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    const Ctor: typeof AudioContext =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    this.context = new Ctor();
    if (this.context.state === "suspended") await this.context.resume();

    this.source = this.context.createMediaStreamSource(this.stream);
    this._analyser = this.context.createAnalyser();
    this._analyser.fftSize = 1024;
    this._analyser.smoothingTimeConstant = 0.75;
    this.source.connect(this._analyser);

    // A zero-gain sink keeps the capture node pulling audio without echoing the
    // mic back to the speakers.
    this.sink = this.context.createGain();
    this.sink.gain.value = 0;
    this.sink.connect(this.context.destination);

    this.blocks = [];
    const onSamples = (samples: Float32Array) => {
      if (this._recording) this.blocks.push(samples);
    };

    let workletReady = false;
    if (this.context.audioWorklet) {
      try {
        const blob = new Blob([WORKLET_SOURCE], { type: "application/javascript" });
        const url = URL.createObjectURL(blob);
        await this.context.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);
        this.worklet = new AudioWorkletNode(this.context, "capture-processor");
        this.worklet.port.onmessage = (event) => onSamples(event.data as Float32Array);
        this.source.connect(this.worklet);
        this.worklet.connect(this.sink);
        workletReady = true;
      } catch {
        workletReady = false;
      }
    }

    if (!workletReady) {
      // ScriptProcessorNode fallback (deprecated but universally available).
      this.processor = this.context.createScriptProcessor(4096, 1, 1);
      this.processor.onaudioprocess = (event) => {
        onSamples(event.inputBuffer.getChannelData(0).slice(0));
      };
      this.source.connect(this.processor);
      this.processor.connect(this.sink);
    }

    this._recording = true;
  }

  /** Stop capture, tear down the graph, and return PCM16 @ targetSampleRate. */
  async stop(): Promise<Int16Array> {
    this._recording = false;
    const inputRate = this.inputSampleRate;

    const totalLength = this.blocks.reduce((sum, b) => sum + b.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const block of this.blocks) {
      merged.set(block, offset);
      offset += block.length;
    }
    this.blocks = [];

    const resampled = resample(merged, inputRate, this.targetSampleRate);
    const pcm = floatToPcm16(resampled);

    await this.teardown();
    return pcm;
  }

  /** Abort capture without producing audio (e.g. user cancelled). */
  async cancel(): Promise<void> {
    this._recording = false;
    this.blocks = [];
    await this.teardown();
  }

  /** Current RMS level in [0,1], for driving a mic meter. */
  level(): number {
    if (!this._analyser) return 0;
    const buffer = new Uint8Array(this._analyser.fftSize);
    this._analyser.getByteTimeDomainData(buffer);
    let sumSquares = 0;
    for (let i = 0; i < buffer.length; i++) {
      const centered = (buffer[i] - 128) / 128;
      sumSquares += centered * centered;
    }
    return Math.min(1, Math.sqrt(sumSquares / buffer.length) * 2.2);
  }

  private async teardown(): Promise<void> {
    try {
      this.worklet?.disconnect();
      this.processor?.disconnect();
      this.source?.disconnect();
      this._analyser?.disconnect();
      this.sink?.disconnect();
      this.stream?.getTracks().forEach((track) => track.stop());
      if (this.context && this.context.state !== "closed") await this.context.close();
    } catch {
      /* teardown is best-effort */
    }
    this.worklet = null;
    this.processor = null;
    this.source = null;
    this._analyser = null;
    this.sink = null;
    this.stream = null;
    this.context = null;
  }
}

/** Split a PCM buffer into WebSocket-sized binary frames (bytes). */
export function pcmFrames(pcm: Int16Array, frameBytes = 32768): ArrayBuffer[] {
  const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  const frames: ArrayBuffer[] = [];
  for (let offset = 0; offset < bytes.length; offset += frameBytes) {
    frames.push(bytes.slice(offset, offset + frameBytes).buffer);
  }
  return frames;
}
