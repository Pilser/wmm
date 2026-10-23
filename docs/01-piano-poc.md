# Piano PoC — listen to pitch, play it back

**Goal:** on a CPU, prove that we can (1) detect the pitch being played, (2) turn it into
MIDI, and (3) generate a piano response that follows that pitch, with acceptable latency.

**Non-goals:** audio generation, polyphonic transcription at first, a trained model at
first, a polished UI.

---

## 1. The signal chain

```
acoustic piano                    digital piano
      │ mic                            │ USB-MIDI
      ▼                                ▼
 audio (44.1 kHz)                 exact note events
      │                                │
      ▼                                │
 pitch tracker (CREPE/pYIN)            │
      │                                │
      ▼                                │
 note segmenter ───────────────────────┤
      │                                │
      ▼                                ▼
        last N notes + current pitch + beat/key
                       │
                       ▼
              small GRU or rules
                       │
                       ▼
             MIDI note events (pitch, velocity, Δt)
                       │
                       ▼
        Web MIDI → DAW/VST piano   or   Tone.js synth
```

The model never touches audio samples. It reads **notes** and writes **notes**. The piano
voice is chosen in the DAW, VST or soundfont.

---

## 2. Which input do you have?

This is the most important practical decision.

| Input | Accuracy | Latency | Cost | Notes |
|---|---|---|---|---|
| **MIDI file (no instrument needed)** | exact pitch, velocity, timing | offline | zero | **best first PoC**; replay timestamps to simulate real time |
| **Digital piano via USB-MIDI** | exact pitch, velocity, timing | ~1–5 ms | zero transcription | best live input; skips the hard part |
| **Acoustic piano via mic + monophonic pitch tracker** | good for one note at a time | ~30–60 ms | cheap (CREPE/pYIN) | octave errors, overtones |
| **Acoustic piano via mic + Basic Pitch** | polyphonic, chords | ~100–200 ms | heavier (ONNX CNN) | best audio-only option |
| **Chroma only** | key/chord, no exact notes | ~20 ms | cheapest | good for accompaniment, not unison |

If you have a digital piano with USB-MIDI, **use it**. The keyboard already solved pitch
detection for us. Audio pitch detection is only needed for an acoustic piano or a recording.

### 2.1 No piano yet? Start from MIDI files

A MIDI file already contains the exact note numbers, velocities, onsets and durations. That
is the same event stream a digital piano would send. So the PoC can start with **zero
hardware**:

1. Parse a piano MIDI file into note events.
2. Replay the events with their timestamps to simulate a live performance.
3. Generate a response (unison, harmony, echo, then a learned model).
4. Write a new MIDI file and render it to audio with a piano soundfont.

This proves the listen → decide → play loop without a microphone. It also gives us
**ground truth**: we know exactly which notes were played, so we can measure whether the
model follows them.

**Datasets (piano):**

| Dataset | Size | Content | License |
|---|---|---|---|
| [MAESTRO v3 MIDI-only](https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip) | 56 MB, 199 h, 7.04M notes | real virtuosic piano performances, velocities + pedals; train/val/test split | CC BY-NC-SA 4.0 |
| [GiantMIDI-Piano](https://github.com/bytedance/GiantMIDI-Piano) | 193 MB, 10,855 files | classical piano, transcribed from recordings | CC BY 4.0 |
| [Lakh MIDI](http://colinraffel.com/projects/lmd/) | 176,581 files | multi-genre, noisy, mostly arranged rather than performed | CC BY 4.0 |
| [ASAP](https://github.com/fosfrancesco/asap-dataset) | classical | aligned scores + performances | check repo |

**Recommendation:** MAESTRO MIDI-only first. It is small, expressive, piano-only, and has a
proper split. 7M notes is enough to train a first small model in minutes, not days.

**Testing the audio listener without a piano:** render a MIDI file to WAV, run the pitch
tracker on that WAV, and compare the detected notes to the known MIDI. This validates the
microphone path offline, with perfect ground truth.

**Rendering MIDI to audio (Linux):**

```bash
sudo apt install fluidsynth fluid-soundfont-gm
fluidsynth -ni -g 1.0 /usr/share/sounds/sf2/FluidR3_GM.sf2 input.mid -F output.wav -r 44100
```

---

## 3. Audio → pitch → MIDI, concretely

### 3.1 Pitch detection

- **CREPE** (monophonic, neural): outputs frequency in Hz + confidence per frame.
- **pYIN** (monophonic, classical): cheaper, less accurate on noisy piano.
- **Basic Pitch** (polyphonic, neural): outputs note events directly, including onset,
  offset, pitch and confidence.

### 3.2 Frequency → MIDI note number

```
midi = 69 + 12 * log2(f / 440)
cents = 1200 * log2(f / f_midi)
```

Examples: 440 Hz → 69 (A4); 261.63 Hz → 60 (C4); 880 Hz → 81 (A5).

Round to the nearest semitone for a note event. Keep `cents` if you want pitch bend or
expressive tuning.

### 3.3 Note segmentation

A pitch tracker gives a stream of frames, not notes. Turn it into notes:

- Stable rounded pitch for ≥ 3 frames (≈30 ms) and confidence > 0.8 → **Note On**.
- Pitch changes by > 0.5 semitone, or confidence drops → **Note Off**, then a new Note On.
- Minimum note length 50 ms; hysteresis to avoid flicker.
- Velocity from onset RMS: map 0–1 to 1–127.
- Timestamp everything against the audio clock, not wall-clock time.

### 3.4 MIDI messages

MIDI is bytes, not sound. The three-byte Note On message:

```
status byte        pitch        velocity
0x90 | channel     0–127        1–127
```

Examples:

```
90 3C 64   Note On,  channel 1, middle C (60), velocity 100
80 3C 00   Note Off, channel 1, middle C (60)
90 3E 50   Note On,  channel 1, D4 (62), velocity 80
```

In the browser:

```js
const out = await navigator.requestMIDIAccess();
const port = [...out.outputs.values()][0];
port.send([0x90, 60, 100]);        // note on, middle C
port.send([0x80, 60, 0], performance.now() + 500); // note off in 500 ms
```

Without a DAW, use Web Audio:

```js
const synth = new Tone.Sampler({ urls: { C4: "C4.mp3" } }).toDestination();
synth.triggerAttackRelease("C4", "8n", Tone.now(), 0.8);
```

The sampler/soundfont supplies the piano timbre. The model only supplies `C4`, duration and
velocity.

---

## 4. Generation: "play on that pitch"

Four levels, cheapest first:

1. **Unison** — output the detected pitch, same octave. Proves the loop.
2. **Octave / fifth / third** — transpose by a fixed interval, constrained to the detected
   key. Sounds musical immediately.
3. **Echo** — repeat the last phrase after a delay, with small variations.
4. **Learned** — a 5M GRU conditioned on the last 16 detected notes + current pitch +
   key/beat, trained on piano MIDI. It learns style, not just rules.

Do 1–3 before training anything. They are 20 lines of code and they prove whether the
latency and tracking feel right. Then replace the rules with the GRU and compare.

---

## 5. Milestones and success criteria

| Stage | Build | Success | Time |
|---|---|---|---|
| 0 | MIDI file → events → render | notes parse correctly; audio plays | 1 day |
| 1 | Rule-based follow (unison/octave/third/echo) | sounds musical; follows the input pitch | 1 day |
| 2 | Simulated real time | replay timestamps; p99 per-event compute < 10 ms | 1–2 days |
| 3 | 5M GRU follow | blind test beats rules; p99 < 50 ms | 1–2 weeks |
| 4 | Live input (USB-MIDI or mic + pitch tracker) | exact notes / stable pitch; <50 ms | 1–2 weeks |
| 5 | Polyphonic (Basic Pitch) | chords detected; latency < 200 ms | 1–2 weeks |

**Kill criteria:** if pitch tracking has frequent octave errors on your piano, switch to
MIDI input or Basic Pitch before training anything. If p99 latency > 100 ms, fix the audio
buffer before adding a model.

---

## 6. Known failure modes

- **Octave errors** — very common in monophonic pitch tracking. Add an octave-correction
  heuristic and use confidence.
- **Piano overtones and sustain** — a single piano note contains many harmonics; simple
  trackers may jump. Test single notes first, then chords with Basic Pitch.
- **Background noise** — add a noise gate / RMS threshold.
- **Web MIDI support** — Chrome/Edge first; Safari is partial.
- **Clock drift** — schedule against the audio clock; do not accumulate `setTimeout`.
- **Latency vs frequency resolution** — a longer FFT window gives better low-note accuracy
  but more latency. 1024 samples at 44.1 kHz ≈ 23 ms is a reasonable start.

---

## 7. What the first commit will contain

- `web/` — AudioWorklet feature extractor + pitch display + Web MIDI out.
- `web/` — Tone.js piano for a self-contained demo.
- `model/` — ONNX export stub (rules first, GRU later).
- `train/` — piano MIDI data pipeline (Phase 1).
- `docs/` — this file and the glossary.

Nothing in `train/` is needed for Stages 0–2. The PoC can be browser-only.
