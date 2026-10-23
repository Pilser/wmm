# Glossary — every abbreviation, in plain language

No separate search needed. If a term is missing, ask and it gets added.

## Project words

- **PoC — Proof of Concept.** A small, deliberately incomplete build that proves the
  riskiest assumption works. Not a product, not pretty. It answers "can this work at all?"
  before we invest months.
- **WMM — World Music Model.** This project: a model that listens to music and generates
  symbolic notes/control, not audio.
- **MVP — Minimum Viable Product.** The smallest version that is actually useful to a user.
  Comes after the PoC.

## Neural networks

- **GRU — Gated Recurrent Unit.** A small recurrent neural network. It reads one input per
  step and keeps a **hidden state** (a list of numbers) summarizing the past. Gates decide
  what to keep, forget and update. Cheap for streaming: constant compute per step, no
  attention over the whole history. Notochord uses one for <10 ms latency.
- **RNN — Recurrent Neural Network.** A network that carries state from one step to the
  next. A GRU is a modern RNN.
- **Transformer.** A network built on **attention**: every position can look at every other
  position. Powerful, but cost grows with sequence length.
- **Attention.** A mechanism that weighs how much each past item matters for the current
  prediction.
- **KV cache — Key-Value cache.** Stored attention keys/values so a transformer does not
  recompute the past for every new token. It saves compute but costs memory.
- **Parameter.** A learned number in the model. "30M params" = 30 million numbers.
- **Token.** A unit the model reads or writes: a word, a note event, a frame.
- **Embedding.** A vector of numbers representing a discrete thing (a pitch, an instrument,
  a word).
- **Hidden state.** The model's internal memory vector between steps.
- **Checkpoint.** A saved set of model parameters.
- **Inference.** Running a trained model to get output.
- **Training.** Adjusting parameters so the model predicts the data better.
- **Teacher forcing.** During training, feeding the model the *real* previous event instead
  of its own guess. Faster and more stable; can cause "exposure bias" at inference.
- **Autoregressive (AR).** Generate one token at a time, each conditioned on the past.
- **Diffusion.** Generate by starting from noise and refining it step by step.
- **Masked diffusion.** Text/music version: "noise" means replacing tokens with `[MASK]`;
  the model learns to fill them in. Used by Mercury, LLaDA.
- **dLLM — diffusion Large Language Model.** An LLM trained with a diffusion objective.
- **LLM — Large Language Model.** A big transformer trained on text.
- **Denoising step.** One refinement pass in diffusion. Fewer steps = faster, usually lower
  quality.
- **NLL — Negative Log-Likelihood.** A training/evaluation score. Lower is better. It
  measures how surprised the model is by real data; it is **not** a musicality score.
- **Quantization.** Two different meanings: (1) storing weights in fewer bits (FP16, INT8);
  (2) rounding time/pitch onto a grid. Context makes it clear.
- **Seed.** A number that makes random sampling reproducible.

## Audio and music

- **MIDI — Musical Instrument Digital Interface.** A protocol of messages (note on, note
  off, pitch, velocity, control change), **not sound**. The sound is made by a synthesizer,
  sampler or DAW.
- **Note On / Note Off.** MIDI messages that start/stop a note. Note On with velocity 0 is
  often used as Note Off.
- **Velocity.** 1–127; roughly how hard a key was struck. Controls loudness and timbre.
- **Pitch / MIDI note number.** 0–127. 60 = middle C (C4); 69 = A4 = 440 Hz.
- **CC — Control Change.** A MIDI message for knobs, pedals, expression (e.g. CC64 = sustain
  pedal).
- **Pitch bend.** A MIDI message that bends pitch continuously, in semitone fractions.
- **MPE — MIDI Polyphonic Expression.** MIDI extension for per-note bend/aftertouch, used by
  expressive controllers.
- **DAW — Digital Audio Workstation.** Software for recording, sequencing and mixing
  (Ableton Live, Logic, Reaper, FL Studio, Bitwig).
- **VST / AU — Virtual Studio Technology / Audio Unit.** Plugin formats for instruments and
  effects inside a DAW.
- **Soundfont / sampler.** A library of recorded instrument sounds (e.g. a piano) played by
  MIDI.
- **Timbre.** The "colour" of a sound — what makes a Steinway differ from a Rhodes. The
  model does **not** generate timbre; the sound module does.
- **Monophonic / polyphonic.** One note at a time / many notes at once.
- **Onset.** The moment a note starts.
- **Chroma.** 12 numbers representing how much energy is in each pitch class (C, C#, D…),
  regardless of octave. Cheap way to estimate key/chords.
- **RMS — Root Mean Square.** A simple measure of loudness/energy.
- **DSP — Digital Signal Processing.** Classic math on audio (filters, FFT, onset detection)
  — cheap and deterministic.
- **FFT — Fast Fourier Transform.** Converts a block of audio samples into frequencies.
- **STFT — Short-Time Fourier Transform.** FFT over short overlapping windows; gives a
  spectrogram.
- **Spectrogram.** A time × frequency map of energy.
- **Sample rate.** Samples per second (44,100 for CD audio).
- **Buffer size.** How many samples are processed at once. Smaller = lower latency, more CPU
  overhead. 128–256 samples ≈ 3–6 ms.
- **Latency.** Delay between action and sound.
- **p50 / p99.** 50th/99th percentile. p99 is the "bad case" latency; averages hide it.
- **CREPE.** A neural pitch tracker (monophonic, frame-by-frame, accurate).
- **pYIN.** A classical probabilistic pitch tracker (monophonic, cheap).
- **Basic Pitch.** Spotify's lightweight polyphonic audio-to-MIDI model (ONNX, CPU).
- **HMM — Hidden Markov Model.** A small classical sequence model; useful as a cheap
  baseline.
- **WAV / MP3.** Audio file formats. MIDI is not audio.

## Web and runtime

- **Web MIDI API.** Browser API to send/receive MIDI messages.
- **Web Audio API.** Browser API for audio processing and synthesis.
- **AudioWorklet.** Runs custom audio code in the browser's audio thread (low latency).
- **WASM — WebAssembly.** Near-native code in the browser; used to run models fast.
- **ONNX — Open Neural Network Exchange.** A standard model file format.
- **ONNX Runtime.** Software that runs ONNX models on CPU/GPU/browser.
- **OSC — Open Sound Control.** Like MIDI, but over a network; common in research/live
  coding (SuperCollider, TidalCycles).
- **WebSocket.** A persistent two-way connection between browser and a local server.

## Compute and cost

- **FLOPs — Floating-Point Operations.** A measure of math work. Training ≈ `6ND`;
  inference ≈ `2N` per token (N = params, D = tokens).
- **MFU — Model FLOPs Utilization.** What fraction of a GPU's peak math you actually use.
- **GPU-hour.** One GPU for one hour; the unit of cloud compute cost.
- **H100 / V100.** NVIDIA data-center GPUs (H100 is newer/faster).
- **HBM.** High-bandwidth memory on data-center GPUs.
- **Memory-bandwidth-bound.** The GPU waits for data instead of doing math. Autoregressive
  decoding is usually in this regime.
