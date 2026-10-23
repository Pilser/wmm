# Rust stack — can Rust do this without Python?

**Short answer: yes, for this project.** Rust is not competitive for training frontier
LLMs, but a 1–30M parameter GRU on 7M note events is a small-model problem. Rust has
everything we need, and it gives us a single binary, no virtualenv, no PyTorch, and a clean
path to WASM.

All versions below were checked on **22 Oct 2026** from crates.io / docs.rs.

---

## 1. Where Rust is mature, and where it is not

| Area | Maturity | For WMM |
|---|---|---|
| Frontier LLM training (100B+) | low | irrelevant |
| Small-model training (1–50M) | medium | **Candle / Burn are enough** |
| Inference (CPU, WASM) | high | **tract / Candle** |
| MIDI file read/write | high | **midly** |
| Live MIDI I/O | high | **midir** |
| Audio I/O and DSP | medium | **cpal + rustfft** |
| Pitch/beat tracking crates | low (stale) | implement YIN/onset ourselves, or run ONNX |
| Notebook-style exploration | low | use CLI commands + CSV logs |
| Web/WASM | high | **wasm-bindgen + tract** |

The gaps are tooling and examples, not capability. Expect **2–3× more code** than the
equivalent PyTorch script, and no Jupyter. For a project this size, that is a fair trade.

---

## 2. The crates

### Training and tensors

| Crate | Version | Role |
|---|---|---|
| `candle-core` | 0.11.0 (Sep 2026) | tensors, autograd, CPU/CUDA/Metal |
| `candle-nn` | 0.11.0 | layers, **GRU**, LSTM, attention |
| `burn` | 0.22.1 (Oct 2026) | full training framework, alternative |
| `burn-nn` | 0.22.1 | layers, **LSTM/GRU**, transformers |

Candle's `candle_nn::rnn::GRU` has `new`, `zero_state`, `step`, `seq`, `seq_init` — exactly
what a streaming model needs. Candle also writes `safetensors`, so checkpoints are portable.

### MIDI and audio

| Crate | Version | Role |
|---|---|---|
| `midly` | 0.5.3 | read/write Standard MIDI Files; fast, zero-alloc |
| `wmidi` | 4.0.11 | live MIDI message parsing |
| `midir` | 0.11.1 | cross-platform real-time MIDI I/O |
| `rustysynth` | 1.3.8 (Oct 2026) | **pure-Rust SoundFont synth: MIDI → audio** |
| `cpal` | 0.18.2 | audio input/output (live path later) |
| `rustfft` | 6.4.1 | FFT for chroma/onset/pitch later |
| `symphonia` | 0.6.1 | decode WAV/MP3/FLAC |

`rustysynth` means we do **not** need FluidSynth or Python to hear the output. Give it a
`.sf2` SoundFont and a MIDI file, get samples.

### Inference and web

| Crate | Version | Role |
|---|---|---|
| `tract-onnx` / `tract` | 0.23.9 (Oct 2026) | pure-Rust ONNX/TF inference, WASM-friendly |
| `ort` | 2.0.0-rc.15 | ONNX Runtime wrapper (fast, but pulls C++ runtime) |
| `wasm-bindgen` | 0.2.129 | Rust ↔ JS |
| `web-sys` | 0.3.106 | Web MIDI / Web Audio bindings |

For the browser PoC, `tract` is the safer choice: pure Rust, no C++ toolchain in the WASM
build. Use `ort` only if native inference speed becomes the bottleneck.

### CLI and plumbing

`clap` 4.6.7, `serde` 1.0.229, `serde_json`, `csv` 1.4.0, `anyhow`, `rand` 0.10.3,
`reqwest` 0.13.5 + `zip` 9.0.5 (or just `curl`/`unzip`).

### Avoid

- **`tch`** — binds libtorch; that is the "large PyTorch" dependency you want to avoid.
- **`dfdx`** — last release 2023; stale.
- **`pitch-detection`** (2022) and **`aubio-rs`** (2021) — stale; implement YIN (about
  30 lines) or run a small ONNX model through `tract`.
- **`rimd`** — last release 2016; use `midly`.

---

## 3. Candle or Burn?

| | Candle | Burn |
|---|---|---|
| Philosophy | minimal, PyTorch-like | batteries-included framework |
| GRU | yes | yes |
| Training loop | you write it | `TrainStep`, metrics, learners |
| Backends | CPU, CUDA, Metal | CPU, CUDA, WGPU, more |
| API stability | moderate | faster churn |
| Best for | a small custom model | a structured training project |

**Recommendation: start with Candle.** It is smaller, has the GRU we need, and keeps the
code close to the math. If the training loop becomes painful (metrics, checkpointing,
schedulers), switch to Burn — the model definition is small enough to port.

---

## 4. Actual crate layout (built)

One binary crate. Files are kept small by rule (see `AGENTS.md`; hard limit 6,000 lines,
target under 800).

```
wmm/
  Cargo.toml
  AGENTS.md
  src/
    lib.rs           # module list, public re-exports
    main.rs          # 11 lines: calls cli::run()
    cli.rs           # clap commands; the only place that parses arguments
    events.rs        # Note, NoteStream, pitch <-> hertz    midi/
      mod.rs         # re-exports
      read.rs        # midly -> NoteStream, note-on/off pairing
      write.rs       # NoteStream -> type-0 SMF at 480 PPQ
      tempo.rs       # ticks <-> seconds, multiple tempo changes
    key.rs           # Krumhansl key detection + diatonic transposition
    harmonize.rs     # unison / octave / third / fourth / fifth / sixth / echo
    render.rs        # rustysynth: MIDI -> stereo 16-bit WAV
    stats.rs         # duration, range, velocity, polyphony, density
    dataset.rs       # MIDI file discovery, MAESTRO split inference
    demo.rs          # built-in A-minor phrase
    model/
      mod.rs         # re-exports
      vocab.rs       # events <-> discrete tokens (time/kind/pitch/velocity)
      corpus.rs      # MIDI files -> fixed-length event windows
      gru.rs         # embeddings + GRU + four prediction heads
      train.rs       # AdamW loop, validation, safetensors checkpoints
      sample.rs      # temperature sampling -> MIDI
      rng.rs         # small deterministic RNG
  tests/
    pipeline.rs      # notes -> file -> notes -> harmony, through public API
```

Commands as built:

```bash
wmm demo --mode third --out demo.mid --render demo.wav
wmm stats data/maestro
wmm harmonize --in in.mid --mode fifth --key "A minor" --out out.mid
wmm render --in out.mid --sf2 assets/piano.sf2 --out out.wav
wmm events --in out.mid --limit 20
```

`stats` accepts files or directories, walks recursively, and has `--json` for logging.
`harmonize` puts the response on a separate MIDI channel (default 1) so a DAW can give it a
different instrument.

Still to add: `download` (asks first), `train`, `sample`.

---

## 5. Compute sanity check

A 1M-parameter GRU on 7M note events:

```
6ND = 6 × 1e6 × 7e6 = 4.2e13 FLOPs
```

On a CPU at ~1e10 effective FLOP/s: **~1 hour**. On a GPU: minutes. The checkpoint is
**~4 MB** in FP32. This is why Rust is enough: the model is smaller than the Rust toolchain.

If the model grows to 30M params and 100M events, `6ND = 1.8e16`, still a few GPU-hours.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Candle's GRU is slower than PyTorch | keep the model ≤5M params; use short sequences; GPU if needed |
| Rust code volume | rules first; add the model only after the loop works |
| MIDI edge cases (tempo maps, multi-track, pedals) | normalize to seconds; test on MAESTRO; keep a debug dump |
| SoundFont licensing | use a permissive small GM `.sf2`; do not bundle a commercial piano |
| No notebook | `wmm stats` + CSV logs + `cargo test` for shapes |
| API churn | pin exact versions in `Cargo.toml`; upgrade deliberately |

---

## 7. First milestone (no model yet)

1. `cargo new wmm`, add `midly`, `clap`, `serde`, `csv`, `anyhow`, `rustysynth`.
2. `wmm stats` on a handful of MAESTRO files: note count, pitch range, duration, tempo.
3. `wmm harmonize` — unison, octave, third/fifth in key, echo.
4. `wmm render` — MIDI → WAV with a SoundFont.
5. Listen. If it sounds musical and follows the input, the data path is proven.

Then add `candle-core` + `candle-nn` and train the GRU.

## 8. Things found while building (22 Oct 2026)

- **Key detection cannot separate a natural minor key from its relative major.** A minor
  and C major contain the same seven notes, so the pitch-class histogram is identical.
  Krumhansl correlation still picks one, but the label is ambiguous. We therefore store
  the demo's key explicitly (`demo::key()`). Good enough for harmony (the scales are the
  same) and a reason to add a bass/final-note heuristic later.
- **`midly` 0.5 has no `u28::MAX` constant.** Variable-length delta times are 28-bit;
  clamp with `0x0FFF_FFFF`.
- **`Smf` has no `to_vec`.** Serialize with `smf.write_std(&mut Vec<u8>)`.
- **rustysynth needs an explicit tail pass.** `MidiFileSequencer::end_of_sequence()` stops
  at the last event, cutting piano release and reverb; we render an extra 1 s of silence
  so notes decay naturally.
