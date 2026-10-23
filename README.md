# WMM — World Music Model

A research project towards a **cheap, real-time, symbolic music model**: it listens to a
live performance and generates notes, timing and control for piano/drums — as MIDI/OSC
events, not audio.

**Start here:** [`docs/00-research-brief.md`](docs/00-research-brief.md)
**Every abbreviation explained:** [`docs/glossary.md`](docs/glossary.md)
**First build:** [`docs/01-piano-poc.md`](docs/01-piano-poc.md)
**Rust stack (no Python):** [`docs/02-rust-stack.md`](docs/02-rust-stack.md)
**Dataset + training budget:** [`docs/03-dataset-and-training.md`](docs/03-dataset-and-training.md)

## Thesis

- **Learn taste, hard-code physics.** Beat, meter, key, range and playability are known.
  Let the model learn style and decisions; let DSP and music theory handle the grid.
- **Two rates, two models.** A tiny fast model at event rate (<10 ms), a small slow model
  at phrase rate (every 1–2 s). Diffusion belongs in the slow loop, not the fast one.
- **Symbols, not samples.** Output MIDI to a DAW/VST/hardware. Timbre is the sound
  module's job. This is the 10–1000× cost saving.
- **Retrieve, vary, sequence.** A motif memory is cheaper and more coherent than asking a
  small network to invent every note.

## Why it can be cheap

| | Text LLM / audio model | Symbolic WMM |
|---|---|---|
| Parameters | 8B–1T | 5M–100M |
| Training data | 1T–15T tokens | 0.1B–5B events |
| Tokens per second of music | 200–44,100 | 10–30 |
| Training compute | 1e23–1e24 FLOPs | ~1e17 FLOPs |
| Training cost | $10k–$100M | $1–$100 |

Anchors: Mittal et al. 2021 trained a symbolic diffusion model in **6.5 h on one V100**;
Notochord runs at **<10 ms per event on a 2017 laptop CPU**; Shazam searched **1.8M tracks
at 1000× realtime** with hashing and a modest server.

## Status

**Working today:** Rust CLI with MIDI read/write, tempo maps, key detection, rule-based
harmony (unison/octave/third/fourth/fifth/sixth/echo), statistics, a pure-Rust MIDI → WAV
renderer, and a trained event GRU (gated recurrent unit) with checkpointing and sampling.
49 unit tests + 3 pipeline tests.

**Data:** MAESTRO v3.0.0 MIDI-only is unpacked in `data/maestro/` and verified —
1,276 files, 7,040,164 notes, 198.65 h, splits 962/137/177, zero parse failures. Parsed
durations match the dataset CSV exactly. See
[`docs/03-dataset-and-training.md`](docs/03-dataset-and-training.md).

**First smoke test:** 204,387 parameters, 10 files, loss 3.88 → 2.31 with validation
tracking (2.46 → 2.32), 800 KB checkpoint, generation and rendering both work.
Next: a 20 h subset run, then compare its output against the rule-based harmony.

## Quick start

```bash
cargo build --release

# Built-in 4-second phrase + diatonic third, rendered to WAV
./target/release/wmm demo --mode third --out demo.mid --render demo.wav

# Inspect any MIDI file or directory
./target/release/wmm stats path/to/file.mid
./target/release/wmm events -i demo.mid --limit 20

# Add a response to an existing performance
./target/release/wmm harmonize -i in.mid -o out.mid --mode fifth --key "A minor"
./target/release/wmm render -i out.mid -o out.wav
```

Train and sample (needs MAESTRO in `data/maestro/`):

```bash
# Smoke test: 10 files, about 6 minutes on a 4-thread laptop
./target/release/wmm train --data data/maestro --files 10 --epochs 2 \
    --out runs/smoke.safetensors

# Generate and listen
./target/release/wmm sample --checkpoint runs/smoke.safetensors \
    --events 300 --out gen.mid --render gen.wav

# Bigger run: 20 h subset, batch 32 is ~1.7x faster than batch 8
./target/release/wmm train --data data/maestro --files 200 --epochs 2 \
    --batch 32 --out runs/small.safetensors
```

Rendering needs a SoundFont (`.sf2`); `wmm` finds a system one automatically, or pass
`--sf2 /path/to/piano.sf2`.

## Non-goals (for now)

- Audio generation / audio diffusion.
- A general-purpose music LLM.
- Cloud inference. The target is a laptop or a pedalboard.
