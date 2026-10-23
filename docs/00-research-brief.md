# WMM — Research Brief 00

**World Music Model: a cheap, real-time, symbolic music model**

Status: research synthesis · Oct 2026 · no code yet

---

## 0. The one-page answer

A WMM that outputs **symbolic control** (MIDI/OSC events), not audio, can be built for
**3–6 orders of magnitude less compute** than a text LLM or an audio diffusion model.
The reason is not a cleverer algorithm. It is that the problem is smaller in every
dimension that costs money:

| Cost driver | Text LLM / audio model | Symbolic WMM |
|---|---|---|
| Model size | 8B–1T params | 5M–100M params |
| Training data | 1T–15T tokens | 0.1B–5B events |
| Tokens per second of content | 50–400 (audio codecs) | 10–30 events |
| Context needed | 128k–1M tokens | 2k–10k events |
| Inference hardware | H100 / HBM | one CPU core, or a laptop GPU |
| Training cost (order) | $10k–$100M | **$1–$100** |

Three concrete anchors from the literature:

- **Mittal et al. 2021** trained a symbolic-music diffusion model (VAE + transformer
  denoiser, 25.6M params) on 170k MIDI files in **6.5 hours on one V100**.
- **Notochord** (2024) runs a GRU-based polyphonic MIDI model at **~6 ms per input event
  and ~3 ms per sampled event on a 2017 laptop CPU** — below 10 ms total latency.
- **Shazam** indexed **1.8M tracks** and searched them at **1000× realtime**, a few
  milliseconds per query, on a modest server — because it turned recognition into
  *hashing and lookup*, not generation.

The expensive part of modern AI is **scale**, not diffusion and not transformers.
A small model on a narrow, well-represented domain is cheap. Music symbols are narrow.

The design thesis:

> **Learn taste. Hard-code physics.**
> Beat, meter, key, range, playability and instrument mapping are known. Don't make a
> network learn them from scratch. Let a small model learn style, timing feel and
> decisions; let DSP and music theory handle the grid.

And the architectural thesis:

> **Two rates, two models.**
> A tiny fast model runs at event rate (~10–30 Hz) and must answer in <10 ms.
> A small slow model runs at phrase rate (every 1–2 s) and plans 8–32 bars.
> Diffusion belongs in the slow loop, not the fast one.

---

## 1. What Mercury actually is

### 1.1 Diffusion for text is not diffusion for images

Image diffusion adds Gaussian noise to continuous pixels and learns to reverse it.
Text is discrete. You cannot add a little Gaussian noise to the word "cat" and get a
meaningful in-between word. So text diffusion uses a different corruption process:

**Masked (absorbing) diffusion.** At noise level `t`, each token is independently either
kept clean or replaced by a `[MASK]` token. At `t=1` everything is masked; at `t=0`
everything is clean. The model is a transformer that sees a partially masked sequence and
predicts the clean tokens at the masked positions. Generation starts from all-masked
noise and iteratively unmasks, coarse-to-fine.

This is the family behind **Mercury** (Inception Labs), **LLaDA**, **SEDD**, **Dream**,
and Gemini's diffusion experiments. Mercury's paper describes exactly this: a transformer
trained on a denoising objective, refining the whole output in parallel, with a
Transformer architecture "orthogonal" to the diffusion algorithm.

### 1.2 Why it is fast

Autoregressive (AR) decoding is **memory-bandwidth-bound**, not math-bound. To produce one
token, the GPU must read every weight in the model once. Arithmetic intensity is about
**1 FLOP per byte**, while an H100 is balanced for ~100–300 FLOP/byte. The math units sit
idle waiting for weights.

Mercury's advantage is that one forward pass updates **many tokens at once**. Weight reads
are amortized over the whole block, arithmetic intensity goes up, and the GPU is used
properly. Inception reports **1109 tokens/s** for Mercury Coder Mini on an H100 and
**0.25 s** average latency on Copilot Arena (fastest in that comparison), versus 27–201
tokens/s for comparable models.
Mercury 2.5 (the latest public release I can verify as of Oct 2026) reports **1107 tok/s**,
260k context, and **$0.20 / $0.75 per million input/output tokens**. If Mercury 3 exists
internally, the mechanism below still applies; I could not verify a public Mercury 3.

Important: this is an **inference** win. Training a dLLM costs roughly the same as training
an AR model of the same size — it is still `6ND` FLOPs over the same tokens. Mercury was
trained on "the order of trillions of tokens" on a large H100 cluster. Diffusion does not
make training cheap. It makes *serving* parallel and predictable.

### 1.3 The honest trade-offs

| | Autoregressive | Masked diffusion |
|---|---|---|
| Training | next-token cross-entropy | masked-token cross-entropy, any order |
| Decode | 1 forward per token, KV cache | K forwards per block, tokens in parallel |
| Latency | grows with output length | roughly flat per block, but K× compute |
| Infilling / constraints | needs FIM or special training | native: clamp tokens, fill the rest |
| Long context | mature KV-cache optimizations | harder; LLaDA explicitly notes no KV cache |
| Best use | streaming, one event at a time | phrase-level planning, editing, repair |

LLaDA (8B, 2.3T tokens) is competitive with LLaMA3 8B, but its authors state that sampling
is *slower* than AR because it uses a fixed context length, cannot yet use KV cache, and
needs roughly as many steps as the response length for best quality. Mercury's speed comes
from a **custom inference engine** (dynamic batching, paged sampling, custom kernels), not
from diffusion alone.

**Lesson for WMM:** do not use diffusion for the per-event real-time loop. Use it (if at
all) to plan a phrase, infill a missing part, or satisfy a set of constraints.

### 1.4 What this means for music

A masked-diffusion model over musical events gives you something genuinely useful:

- **Clamp** the notes on beat 1, or the chord tones, or the drummer's kick pattern.
- **Fill** the missing alto voice, the missing bar, the missing drum fill.
- **Repair** a phrase that drifted out of key.
- **Generate a 4-bar phrase** in 4–8 denoising steps instead of 200 AR steps.

That is a *compositional* tool. It is not a live instrument. The live instrument needs a
causal model with constant latency.

---

## 2. What actually makes LMs and diffusion models expensive

### 2.1 The two formulas

Training compute for a transformer:

```
C_train ≈ 6 · N · D
```

`N` = parameters, `D` = training tokens. The 6 is roughly: 2 FLOPs per parameter per token
forward, 4 backward.

Inference compute:

```
C_infer ≈ 2 · N  FLOPs per token   (plus attention)
```

Memory per token of KV cache:

```
KV_bytes ≈ 2 · layers · kv_heads · head_dim · seq_len · bytes_per_value
```

For an 8B model with 32 layers, 8 KV heads, head_dim 128, 32k context, FP16:
`2 × 32 × 8 × 128 × 32768 × 2 ≈ 4.3 GB` — just for the cache.

### 2.2 The real cost stack

1. **Data scale.** `D` dominates. Trillions of tokens is the single biggest multiplier.
2. **Parameter count.** `N` multiplies training and sets the memory floor at inference.
3. **Sequence length.** Attention is `O(L²)` in training; KV cache is `O(L)` in memory.
4. **Memory bandwidth.** AR decode reads all weights per token; GPUs idle. This is why
   batching and parallel decoding matter more than raw FLOPs.
5. **Diffusion steps.** Each sample costs `K` forward passes. `K=1000` (old image models)
   is brutal; `K=4–8` is fine. Mittal's 2021 music model used 1000 steps — modern few-step
   samplers remove most of that.
6. **Infrastructure.** Interconnect, serving, reliability, evaluation, data curation.
   The GPU bill is often the smaller half of a real project.
7. **Modality entropy.** Audio has 16k–44.1k samples per second; a neural codec still
   produces 200–400 tokens per second. MIDI produces 10–30 *events* per second.

### 2.3 Rough comparison

Assumptions are deliberately round; use these for orders of magnitude, not accounting.

| System | N | D | `6ND` | Rough GPU-hours | Note |
|---|---|---|---|---|---|
| GPT-3 175B | 175B | 300B | 3.2e23 | ~1–2M V100 | $2–4M at 2020 prices |
| LLaDA 8B | 8B | 2.3T | 1.1e23 | ~80k H100 | masked diffusion, AR-comparable |
| Mercury-class dLLM | not public | trillions | ~1e23–1e24 (est.) | 100k+ H100 | inference-optimized |
| Audio model, 1.5B | 1.5B | ~1.4e10 codec tokens | ~1e20 | 10²–10⁴ | plus codec training |
| Mittal symbolic diffusion | 25.6M | ~1e9 latent tokens processed | ~1.5e17 | **6.5 (reported)** | one V100, 2021 |
| **WMM (proposed)** | **30M** | **1e9 events** | **1.8e17** | **~3–10** | CPU inference |

The Mittal row is the important one: the formula underestimates small-model overhead, but
the *reported* number is 6.5 hours on one V100 for a working symbolic diffusion model.
That is a laptop-scale research project. The gap to GPT-3 is roughly **1.8 million×** in
training compute (about six orders of magnitude).

### 2.4 The token-density argument

Per second of music:

| Representation | Tokens / second | 4-minute piece |
|---|---|---|
| Raw audio 44.1 kHz mono | 44,100 | 10.6M samples |
| Neural codec 50 Hz × 4 codebooks | 200 | 48k tokens |
| MIDI events (piano + drums) | 10–30 | 2.4k–7.2k events |
| MIDI subtokens (REMI-style) | 40–120 | 10k–29k tokens |

Audio is **10–1000× denser** than symbols. That single fact explains most of the cost gap.
It is also why the model should output symbols and let a DAW, VST piano, or hardware sound
module handle timbre. You are building the **brain**, not the **body**.

---

## 3. Cheap world models: what Dreamer teaches

A "world model" in the Dreamer sense is three networks:

1. **World model**: `p(s_t | s_{t-1}, a_t)` plus an observation decoder.
2. **Critic**: value of a state.
3. **Actor**: policy that maximizes value *inside the learned model*.

DreamerV3 does this with **8M–200M parameters**, one V100 per agent, fixed hyperparameters,
and learns Minecraft diamonds from scratch. It is cheap because the latent state is small,
rollouts are short (horizon 16), and the policy is trained in imagination rather than in the
real environment.

A 2026 reproduction of Dreamer 4 on CoinRun trained a **1.57B-parameter dynamics model** on
9.6M frames for about **$150** on one H200 (plus a tokenizer run of 1h35m on an H100). That
is a real video world model for the price of a dinner. The lesson is not "1.57B is small";
it is that **a narrow, synthetic, well-tokenized domain is cheap**, while open-world video
is expensive because the data and the tokenizer are expensive.

### 3.1 The cheap-world-model recipe

1. **Compress hard.** A 64–256 dim latent, not pixels or waveforms.
2. **Learn the dynamics, not the physics.** Hard-code what you know.
3. **Short horizon.** 8–32 steps of imagination is enough for most decisions.
4. **Action-condition.** The model must answer "what happens if I do X?"
5. **Skip RL at first.** Self-supervised next-state prediction gives you most of the value.
6. **Evaluate on controllability, not just reconstruction.** A model that predicts well but
   ignores actions is useless for interaction.

### 3.2 Mapping to music

| World model | WMM |
|---|---|
| observation `o_t` | last N events + audio-derived features |
| latent state `s_t` | beat phase, key, energy, texture, gesture, motif memory |
| action `a_t` | what the performer/controller does (note, bend, CC, pedal, knob) |
| dynamics `s' = f(s, a)` | "if I play this, what does the ensemble do next?" |
| reward | optional: stay in key, lock to grid, be novel, be playable |
| imagination | roll out 2–8 bars, score, choose |

The research contribution is **action-conditioned counterfactuals**: "what if the drummer
plays halftime?", "what if I hold this chord for two bars?", "what if I want the model to
answer my phrase but not copy it?" A plain MIDI generator cannot answer those. A world
model can, because it separates state from action.

But do not start with RL. Start with a next-event model that is *conditioned on the
performer's stream*. That is already a world model in the useful sense; the actor/critic
can come later if you need high-level decisions (when to enter, how dense to be).

---

## 4. Shazam: the cheap-compute blueprint

### 4.1 How it works

1. **STFT** the audio into a time–frequency energy map.
2. **Pick local peaks**: a bin is a peak if it is louder than its neighborhood. This throws
   away amplitude and keeps only robust coordinates. The sparse scatter plot is the
   "constellation map".
3. **Pair peaks**: for each anchor peak, pair it with peaks in a target zone (a later time,
   a nearby frequency band). Each pair gives `(f1, f2, Δt)` — three numbers with much higher
   entropy than a single peak.
4. **Hash**: pack the pair into a **32-bit hash**. Store `(hash, track_id, time_offset)` in a
   sorted index.
5. **Query**: hash the sample's pairs, look them up, and histogram the `(track, offset)`
   differences. A correct match produces a spike on a diagonal; noise produces a flat
   histogram.

### 4.2 The numbers

- **1.8M tracks** in the 2003 paper (nearly 2M mentioned; 3M by 2006).
- **1000× realtime** search; **a few milliseconds per query** for radio monitoring.
- 2006 deployment: a cluster of **a few dozen commodity 64-bit x86 servers, 8 GB RAM each**.
- Combinatorial pairs trade **~10× storage** for **~10,000× search speed**.

Back-of-envelope check: 1.8M tracks × 4 min ≈ 432M track-seconds. At ~30 hashes/s that is
~1.3e10 hashes × 8 bytes ≈ **104 GB**. Add index overhead and you land at a few hundred GB
— exactly "a few dozen 8 GB servers". The arithmetic is consistent, and it shows how far
sparse hashing stretches commodity hardware.

### 4.3 The four lessons for WMM

1. **Sparse, robust features beat dense, fragile ones.** Peaks survive noise, EQ and codecs.
   For music state, use beat phase, onset strength, chroma/key, tempo, energy — not raw audio.
2. **Pairs raise entropy cheaply.** One feature is ambiguous; a *relation between two
   features* is specific. The same trick works for motifs: a motif is a relation between
   intervals, not a list of absolute pitches.
3. **Turn generation into retrieval where you can.** Keep a memory of phrases, grooves,
   licks and transitions. Let the model *select, vary and sequence* them instead of
   inventing every note from a blank latent. This is how you get coherence cheaply.
4. **Align in time, then histogram.** Shazam does not match point sets; it matches
   *offsets*. A live model should track a beat-phase offset and a key offset, and correct
   them continuously — a phase-locked loop, not a re-inference from scratch.

### 4.4 What Shazam is *not*

Shazam identifies a recording. It does not generate, and it does not handle performance
variation (it assumes linear time). A WMM needs *state tracking under variation*: tempo
drift, expressive timing, wrong notes, improvisation. The Shazam analogy is about **cheap
front-end features + hashed memory + temporal alignment**, not about copying the algorithm.

---

## 5. What already exists in symbolic music

### 5.1 Mittal et al., "Symbolic Music Generation with Diffusion Models" (2021)

- Two-stage: a pretrained **MusicVAE** compresses 2-bar phrases into 42-dim latents; a
  **6-layer transformer** denoiser models 32 latents (64 bars).
- 25.58M trainable params. Trained on **Lakh MIDI** (170k files, ~989k 64-bar monophonic
  sequences) for 500k updates, **6.5 hours on one V100**.
- Non-autoregressive, constant number of refinement steps, post-hoc infilling by clamping
  known regions and re-noising them at each step.
- Beat a Transformer-MDN AR baseline on framewise pitch/duration self-similarity.

**Takeaway:** a working symbolic diffusion model is a one-GPU project. The hard part is
representation and evaluation, not compute.

### 5.2 Notochord (2024)

- Event tuples: `(Δt, velocity, pitch, instrument)`, with note-on/note-off as separate
  events, **continuous time in seconds** (not quantized), and a discretized mixture of
  logistics for time and velocity.
- **GRU** backbone + per-sub-event MLPs, trained on Lakh MIDI. Any-order sub-event
  factorization lets the user clamp pitch, time, velocity or instrument independently.
- **~6 ms to feed an event, ~3 ms to sample one**, on a 2017 MacBook Pro i7. Sub-10 ms
  total. Runs in Python + OSC, with SuperCollider/Tidal front-ends.
- Explicitly not state-of-the-art in coherence; explicitly designed for **interaction**.

**Takeaway:** real-time symbolic generation is solved at small scale. The remaining
problems are musical quality, long-range structure and control.

### 5.3 Anticipatory Music Transformer (2023)

- Interleaves **event tokens** and **control tokens** so controls can appear *after* the
  events they constrain. A control token carries a δ-second lookahead.
- No architecture change: it is a GPT-2-style causal transformer with a modified data
  layout and doubled vocabulary.
- Enables infilling and accompaniment with asynchronous control, and matches AR baselines
  on unconditional/prompted generation.

**Takeaway:** controllability can come from **data layout**, not a new architecture. This is
the cheapest possible way to get "fill in the missing part" behavior.

### 5.4 Tokenization: the choice that decides your cost

| Representation | Length for 4 min | Pros | Cons |
|---|---|---|---|
| REMI / MidiTok | 10k–30k tokens | mature, downbeat/bar structure | long, quantized, loses microtiming |
| Event tuples (Notochord) | 2k–7k events | short, continuous time, expressive | custom code, no off-the-shelf tokenizer |
| Piano roll frames | 100–400 per second | simple, image-like | huge, quantized, wasteful on silence |
| Audio codec | 48k+ tokens | realistic timbre | 10–100× cost, needs codec |

**Recommendation:** start with **event tuples + continuous time**, Notochord-style. It is the
cheapest representation that preserves groove, and it maps directly to MIDI out.

---

## 6. The WMM architecture I would build

### 6.1 What it is

A **symbolic, action-conditioned, two-rate model** that:

- **listens** to a stream of MIDI/OSC events and control frames,
- **maintains** a small latent state (beat phase, key, energy, texture, motif memory),
- **predicts** the next event distribution and, optionally, the next control state,
- **plans** a phrase every 1–2 seconds,
- **outputs** MIDI/OSC to a DAW plugin, VST piano, drum kit, or hardware.

It does **not** generate audio. Timbre is the sound module's job.

### 6.2 The two rates

```
FAST LOOP  (event rate, 10–30 Hz, <10 ms budget)
  input : last 64–256 events + control frame
  model : GRU or small SSM, 2–10M params
  output: next event (Δt, pitch, vel, inst) + control deltas
  runs  : on CPU, in the audio thread's lookahead window

SLOW LOOP  (phrase rate, every 1–2 s)
  input : latent summary + style/intent + recent motif memory
  model : 20–80M transformer or masked-diffusion planner
  output: a plan for the next 8–32 bars (key, texture, motif, density, entries)
  runs  : on GPU or a spare core, asynchronously
```

The fast loop never waits for the slow loop. The slow loop writes into a **plan buffer**;
the fast loop reads the current plan and realizes it event by event. If the plan is late,
the fast loop keeps using the previous one. This is the same pattern as a game AI: a
high-level planner and a low-level controller.

### 6.3 Event and control schema

```json
{"t": 12.483, "kind": "note_on",  "pitch": 60, "vel": 96, "inst": 1, "ch": 0}
{"t": 12.731, "kind": "note_off", "pitch": 60, "vel": 0,  "inst": 1, "ch": 0}
{"t": 12.500, "kind": "cc",       "cc": 64, "val": 127, "ch": 0}
```

Control frame at 100 Hz (derived from audio and/or MIDI, cheap DSP):

```json
{"t": 12.50, "beat_phase": 0.25, "tempo": 118.2, "key": 0,
 "energy": 0.62, "density": 0.40, "onset": 0.81, "knob1": 0.30}
```

The model input is the event history plus the current control frame. The output is a
distribution over the next event tuple, plus optional control deltas (pitch bend, expression,
pedal, density). "Continuous" here means **control-rate continuity**, not audio samples:
the state updates every 10 ms, even though notes are discrete events.

### 6.4 Why this is a world model

- The latent state is a **compressed observation** of the musical situation.
- The action is the performer's event/control stream.
- The dynamics model predicts the next state and the next event.
- A planner can roll out 2–8 bars under different actions and pick one.
- Auxiliary heads predict **beat phase, key, energy and density**; this forces the latent to
  contain the "physics" rather than memorizing surface tokens. Cheap multi-task trick.

### 6.5 Where diffusion fits

Only in the slow loop, and only if it earns its keep:

- **Phrase infilling**: clamp the first bar and the drummer's kick, fill the rest.
- **Constraint satisfaction**: all notes in key, chord tones on strong beats, range limits.
- **Multiple candidates**: generate 4 plans in parallel, score them, pick one.
- **Repair**: re-denoise a phrase that drifted.

If a small AR planner with anticipation tokens does the job, use that instead. It is simpler
and has a KV cache.

### 6.6 Where retrieval fits

Keep a **motif memory**: a vector index of short phrases (2–8 beats), tagged by genre,
instrument, density, key-relative intervals. The planner retrieves 3–5 candidates and the
fast loop varies them. This is the Shazam lesson applied to generation: don't invent every
note; **retrieve, transpose, vary, sequence**. It is cheap, coherent, and gives the model a
sense of style that a 30M-param net cannot learn from 1B events.

---

## 7. Cost model for WMM

### 7.1 Training

For 30M params and 1B events:

```
C = 6 × 3e7 × 1e9 = 1.8e17 FLOPs
```

At a conservative 2e13 effective FLOP/s on a consumer GPU (25% of ~80 TFLOPS FP16):
`9,000 s ≈ 2.5 GPU-hours`. Add data loading, eval and restarts: **3–10 GPU-hours**.
At $0.5–2/hour: **$2–$20**. Even a 10× mistake keeps it under $200.

For 100M params and 2B events: `1.2e18` FLOPs → **~7–25 GPU-hours**. Still cheap.

### 7.2 Inference

Fast loop, 30M params, 20 events/s:

```
2N per event = 6e7 FLOPs
× 20 events/s = 1.2e9 FLOP/s
```

A modern CPU core does ~1e10–1e11 FLOP/s with AVX. So the math is **1–10% of one core**.
Notochord's 9 ms/event is dominated by Python and framework overhead, not FLOPs. In C++,
ONNX Runtime or RTNeural, sub-millisecond is realistic. Budget **<2 ms p99** for the model
and the rest for scheduling and MIDI I/O.

Slow loop, 64-event phrase, 8 denoising steps, 30M params:

```
8 × 64 × 6e7 ≈ 3.1e10 FLOPs per phrase
÷ 2 s per phrase ≈ 1.5e10 FLOP/s
```

That is one core or a small GPU, comfortably. It is ~10× the fast loop, which is why it runs
asynchronously.

Compare an 8B LLM: `2N = 1.6e10 FLOPs/token`; at 50 tok/s that is `8e11 FLOP/s`, and it must
stream ~16 GB of weights per token from HBM. That is why it needs a data-center GPU and you
don't.

### 7.3 Memory

- 30M params FP16 = **60 MB**. Fits in L3 on some CPUs, easily on a phone.
- Fast-loop state: 256–1024 floats = **1–4 KB**. Negligible.
- Motif memory: 1M phrases × 64 dims × FP16 = **128 MB**. Optional, can be 10k phrases.
- No KV cache in the fast loop if you use a GRU/SSM; a sliding-window transformer keeps it
  bounded.

### 7.4 What would make it expensive

Avoid these:

1. Generating audio end-to-end. (10–1000× more tokens.)
2. A 1B+ transformer at event rate. (Waste; use a planner instead.)
3. 1000-step diffusion. (Use 4–8 steps.)
4. Attention over hours of music. (Use a summary state + retrieval.)
5. Training on raw, uncleaned MIDI. (Garbage in, expensive garbage out.)
6. RL with a hand-written reward before you have a good base model. (Reward hacking.)
7. Python in the audio callback. (Use C++/ONNX/RTNeural.)
8. Baking timbre into the model. (Keep the symbolic/audio boundary.)

---

## 8. What will actually be hard

Compute is not the bottleneck. These are:

1. **Long-range structure.** A 30M model can learn a groove; it cannot learn sonata form.
   The two-rate planner and motif memory are the mitigation.
2. **Groove and expressive timing.** Quantized tokens kill feel. Use continuous Δt and
   velocity, and evaluate with onset histograms, not just NLL.
3. **Data quality and licensing.** Lakh is noisy and biased toward GM instrument 1.
   Groove MIDI is small but expressive. GiantMIDI-Piano is classical only. Check licenses
   before shipping.
4. **Evaluation.** NLL is not musicality. You need blind human A/B tests, plus simple
   statistics: beat alignment, density, pitch-class entropy, repetition rate, range
   violations.
5. **Latency jitter.** Average latency is irrelevant; p99 and jitter are what make an
   instrument feel bad. Measure on the target laptop, not a dev box.
6. **Mode collapse / repetition.** Sampling temperature, nucleus, and a novelty penalty in
   the planner.
7. **Instrument mapping.** GM program numbers are not your piano voices. Keep a mapping
   layer; never bake timbre into the symbolic model.
8. **MIDI 2.0 / MPE.** If you want expressive control (per-note bend, aftertouch), design
   the event schema for it now, even if you only emit MIDI 1.0 at first.

---

## 9. Phased plan (deliberately slow)

Each phase has one artifact, one metric, and one kill criterion. Do not skip the kill
criterion; it is what keeps the project on track.

### Phase 0 — Data and baseline (3–5 days)

- Build the event schema and a cleaning pipeline for Lakh + Groove MIDI (+ GiantMIDI-Piano).
- Deduplicate, remove empty/overlapping notes, normalize tempo, keep continuous Δt.
- Compute corpus stats: events/song, events/sec, pitch/instrument histograms, tempo range.
- Implement a **2nd-order Markov baseline** over event tuples.

**Artifact:** a reproducible dataset + baseline that generates a MIDI file.
**Metric:** events/sec, vocabulary size, baseline NLL/event.
**Kill:** fewer than 10M clean events → narrow the scope (piano-only or drums-only) before
building a neural model.

### Phase 1 — Fast-loop GRU (1–2 weeks)

- 5–10M param GRU + per-sub-event heads, continuous time/velocity (mixture of logistics).
- Train with next-event NLL, teacher forcing, tempo/transpose/velocity augmentation.
- Export to ONNX; measure p50/p99 latency per event on the target laptop CPU.

**Artifact:** `wmm sample --seed 42 --bars 32` writes a MIDI file.
**Metric:** NLL vs Markov; p99 latency.
**Kill:** NLL not clearly better than Markov, or p99 > 10 ms in ONNX → fix data/latency
before adding anything.

### Phase 2 — Real-time loop and DAW bridge (1–2 weeks)

- Lookahead scheduler with a plan buffer; MIDI out via ALSA/CoreMIDI or OSC.
- VST3/AU shell (JUCE) or a standalone app; start with a standalone + virtual MIDI port.
- Play with it: auto-pitch, harmonizer, "answer my phrase", "hold this chord".

**Artifact:** you can improvise with it for 10 minutes without a crash.
**Metric:** p99 latency, dropped-note rate, subjective "feel".
**Kill:** p99 > 20 ms or it feels laggy → switch runtime (C++/RTNeural) before adding
features.

### Phase 3 — Conditioning and control (1–2 weeks)

- Clamp pitch, time, velocity, instrument independently (any-order sub-event factorization).
- Add control-frame inputs: beat phase, key, energy, density, one or two knobs.
- Add auxiliary heads for beat/key/energy prediction.

**Artifact:** a MIDI controller mapping where knobs change density/key/register live.
**Metric:** intervention response time; does clamping actually do what you ask?
**Kill:** interventions feel unresponsive or ignored → check conditioning leakage, retrain
with random sub-event masks.

### Phase 4 — Slow-loop planner (2–4 weeks)

- 20–80M transformer or masked-diffusion planner over 8–32 bar plans.
- Start with AR + anticipation tokens (simpler, KV cache); try diffusion only if you need
  infilling/constraints.
- Score multiple plans with musical constraints; pick one.
- Motif memory: index 2–8 beat phrases, retrieve and vary.

**Artifact:** 32–64 bar generations with audible long-range structure.
**Metric:** blind A/B vs Phase 1; repetition rate; key/range violations.
**Kill:** planner does not improve blind ratings → drop it, keep the fast loop and retrieval.

### Phase 5 — World-model rollouts (optional, research)

- Action-conditioned dynamics: "if the performer does X, what happens next?"
- Roll out 2–8 bars, score, choose (model-predictive control).
- Optional RL only for high-level decisions (entry, density, form), with a constrained
  reward and a strong prior from the supervised model.

**Artifact:** counterfactual demos: halftime, held chord, call-and-response.
**Metric:** controllability, not just plausibility.
**Kill:** rollouts are worse than direct sampling → the dynamics model is not good enough;
go back to supervised data.

### Phase 6 — Personalization (ongoing)

- Fine-tune on your own playing and your library.
- Learn a style embedding (per artist/album/kit) rather than separate models.
- Map to your piano voices and drum kits; never bake timbre into the model.

---

## 10. Reading list

**Diffusion LMs**
- Mercury: *Ultra-Fast Language Models Based on Diffusion* — https://arxiv.org/abs/2506.17298
- Inception blog, Mercury 2.5 — https://www.inceptionlabs.ai/blog/introducing-mercury-2-5
- LLaDA: *Large Language Diffusion Models* — https://arxiv.org/abs/2406.07524
- Diffusion LLMs: training/inference engineering overview —
  https://www.dailydoseofds.com/diffusion-models-part-2

**Symbolic music**
- Mittal et al., *Symbolic Music Generation with Diffusion Models* —
  https://arxiv.org/abs/2103.16091 · code: https://github.com/magenta/symbolic-music-diffusion
- Notochord — https://arxiv.org/abs/2403.12000 · code:
  https://github.com/Intelligent-Instruments-Lab/notochord
- Anticipatory Music Transformer — https://arxiv.org/abs/2306.08620
- MidiTok (tokenizers) — https://github.com/Natooz/MidiTok

**World models**
- DreamerV3 — https://arxiv.org/abs/2301.04104 · code: https://github.com/danijar/dreamerv3
- Dreamer 4 CoinRun reproduction (~$150) — https://github.com/RajatDandekar/dreamer4-coinrun
- *The Cost of Dreaming* (survey of compute constraints in world models) —
  https://ieeexplore.ieee.org/abstract/document/11663588

**Shazam**
- Wang, *An Industrial-Strength Audio Search Algorithm* —
  https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf
- Wang, *The Shazam music recognition service*, CACM 2006 —
  https://dl.acm.org/doi/pdf/10.1145/1145287.1145312

**Cost**
- *The FLOPs Calculus of Language Model Training* —
  https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
- *Optimize for inference too, not just training FLOPs* —
  https://matx.com/research/lifetime_llm_cost

---

## 11. Open questions for the next session

1. **Scope of the first model:** piano only, drums only, or piano + drums? (Piano + drums is
   the smallest useful ensemble and matches your "piano or drums" goal.)
2. **Interface first:** standalone + virtual MIDI port, or straight to a VST3/AU plugin?
   (Standalone is faster to iterate; the plugin is the real product.)
3. **Expression:** MIDI 1.0 only at first, or design for MPE/MIDI 2.0 per-note expression
   from day one? (Design the schema for it now; emit 1.0 first.)

My recommendation: **piano + drums, standalone + virtual MIDI, schema that can carry MPE
later.** Then build Phase 0 this week.
