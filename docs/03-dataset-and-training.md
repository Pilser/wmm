# Dataset check — MAESTRO v3.0.0 (MIDI only)

Verified **22 Oct 2026** by parsing every file with `wmm stats`.

Archive: `data/maestro-v3.0.0-midi.zip` (56 MB)
Unpacked: `data/maestro/` (84 MB, 1,276 files)
License: **CC BY-NC-SA 4.0** — non-commercial, share-alike. Fine for research here;
not for a commercial product.

## 1. Is this the data we need?

Yes. It is exactly the piano event stream the project is designed around.

| Check | Expected | Measured | Verdict |
|---|---|---|---|
| MIDI files | 1,276 | **1,276** | ✅ |
| Note events | 7.04 M | **7,040,164** | ✅ exact |
| Duration | 199 h | **198.65 h** | ✅ |
| Split sizes | 962 / 137 / 177 | **962 / 137 / 177** | ✅ |
| Parse failures | 0 | **0** | ✅ |
| Parsed duration vs CSV | equal | **198.65 h = 198.65 h** | ✅ |

Duration is computed by `wmm` from ticks + tempo map, independently of the CSV, and it
matches the CSV's own duration column to two decimals. That is strong evidence that the
tempo-map and note-pairing code is correct.

Per split:

| Split | Files | Notes | Hours | Mean velocity | Mean polyphony |
|---|---|---|---|---|---|
| train | 962 | 5,659,329 | 159.24 | 64.6 | 1.99 |
| validation | 137 | 639,425 | 19.43 | 65.0 | 2.03 |
| test | 177 | 741,410 | 19.98 | 65.1 | 1.97 |
| **all** | **1,276** | **7,040,164** | **198.65** | **64.7** | **1.99** |

Other facts:

- Pitch range **A0–C8** (MIDI 21–108), the full 88-key piano.
- Notes per file: min 129, median 4,395, max 25,076.
- Note density: median 9.9 notes/s, 99th percentile 20.4 notes/s.
- One instrument (piano) on one channel, in **two tracks** per file (a tempo/metadata
  track and the piano track). Real performance timing (not quantized).
- Default tempo is 120 BPM with tempo-change meta events for rubato; `wmm` reads the whole
  tempo map, which is why our durations match the CSV.
- The files live in year folders (`2004/`, `2008/`, …), **not** in split folders. The split
  comes from `maestro-v3.0.0.csv`, which `wmm` now reads with `--manifest` or auto-detects.

## 2. What is *not* in this download

- **No audio.** We do not need it: the model reads and writes notes, not samples.
- **No sustain pedal in our NoteStream.** MAESTRO does contain CC64 pedal messages, but
  `wmm` currently keeps only note-on/note-off. This is fine for a note-level model; it
  makes rendered audio sound drier than the real performance. Pedal can be added later as
  a control event.
- **No quantization.** Timing is human, with expressive rubato. Good for a "follow the
  performer" model, harder for a "grid" model. We can quantize later if needed.

## 3. Training budget

Two machines were measured on **22 Oct 2026** with the same benchmark (naive 512×512
double matrix multiply, `gcc -O3 -march=native -fopenmp`). Neither has a usable GPU, so
both are CPU-only.

| | Laptop (this repo) | IICO (remote, over SSH) |
|---|---|---|
| CPU | i5-4300U, 2 cores / 4 threads, 1.9–2.9 GHz (Haswell) | i5-12500, 6 cores / 12 threads, up to 4.6 GHz (Alder Lake) |
| Single-thread | 5.8 GFLOP/s | **18.7 GFLOP/s** (3.2×) |
| All threads | 9.6 GFLOP/s (4 threads, thermally limited) | **144.5 GFLOP/s** (12 threads, 15×) |
| RAM | 7 GB, ~4 GB free | 22 GB, ~11 GB available |
| Disk free | 102 GB | 148 GB (NVMe) |
| GPU | Intel HD 4400 (unusable for this) | Intel UHD 770 (unusable for this) |
| Rust toolchain | installed | **not installed — ship the release binary** |
| Other load | light | **busy: mailcow mail server, Supabase, n8n, Firefox** |

The release binary built here runs unmodified on IICO (glibc 2.41 here, 2.43 there), so no
Rust install is needed there: `cargo build --release` locally, copy 2 MB, run.

Cost rule: forward + backward is about `6 × N × D` floating-point operations for `N`
parameters and `D` events (the same 6ND rule used in the research brief).

| Tier | Model | Data | FLOPs / epoch | Laptop | IICO (12 threads) |
|---|---|---|---|---|---|
| Smoke test | ~100 K params | 10 files | 1e10 | < 1 min | < 1 min |
| Small | ~300 K params | 20 h subset | 1.3e12 | 10–20 min | 1–2 min |
| Full | ~300 K params | 159 h train (5.7 M notes) | 1e13 | 1–3 h | **5–15 min** |
| Full, bigger | ~1 M params | 159 h train | 3.4e13 | 4–12 h | 20–60 min |

These assume 10–30 % of the benchmark throughput, which is realistic for small GRU
matrices in Candle. The first real measurement will replace this table.

1 M parameters is the practical ceiling for the laptop; IICO could go somewhat higher.

### First measurements (laptop, 23 Oct 2026)

Smoke test: 10 MAESTRO files, 136,284 events, 528 windows, hidden 128, embed 64,
**204,387 parameters**, two passes.

| Batch | Throughput | Notes |
|---|---|---|
| 8 | 758 events/s | 5 min 40 s for 2 epochs |
| 32 | ~1,300 events/s | 1.7× faster; larger matrices use the CPU better |

Loss fell from **3.88** to **2.31** in 126 steps, and validation loss tracked it
(2.4556 → 2.3155), so the model is learning rather than memorising. At 1,300 events/s,
one pass over the 5.7 M-event train split is **~1.2 h** on this laptop — inside the
1–3 h estimate, but long enough that a 20 h subset is the sensible next run.

The checkpoint is 800 KB (FP32) plus a small `.json` sidecar with the shapes. Sampling 300
events produced an 81-note, 11.4 s MIDI file and a 12.4 s WAV, so train → sample → render
is closed end to end.

### IICO is a shared machine

It runs mailcow (mail), Supabase (databases), n8n-style automations and a browser. Training
must not starve them:

- use `nice -n 10` and `ionice -c 2 -n 7`;
- cap threads, e.g. `--threads 6` instead of 12 (about 2× slower, far less disruptive);
- expect ~0.7 baseline load and ~11 GB RAM already in use.

## 4. Prep before a training run

Only needed for the **full** tier. The smoke test can run while you use the machine.

On the laptop:

1. Plug in the charger. A 4-core laptop will thermally throttle on battery.
2. Close browsers, video calls, games, Docker, virtual machines. They cost RAM and cores.
3. Aim for ≥ 3 GB free RAM (`free -h`). Our batches are small, so this is usually easy.
4. Put the laptop on a hard surface; expect the fan to run for the whole run.
5. Run with `cargo run --release`, never the debug build (10–50× slower).
6. Keep the terminal open; checkpoints go to `runs/`, logs to `runs/train.log`.

On IICO (shared): skip the charger and fan steps, but cap threads (`--threads 6`) and run
under `nice -n 10` so mail and databases keep responding.

## 5. Suggested order

1. **Smoke test** — 10 files, 2 epochs. Proves the loss goes down and the sampler emits
   valid MIDI. No need to close anything.
2. **Small run** — 20 h, ~30 min. Produces something listenable to compare against the
   rule-based harmony.
3. **Full run** — overnight if it is promising.

Do not start with tier 3. Most bugs in a first training loop show up in the first 100
steps, and those cost seconds, not hours.
