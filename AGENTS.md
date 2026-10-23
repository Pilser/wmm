# AGENTS.md — rules for working in this repo

These rules are mandatory for any agent (human or AI) changing this repository.

## 1. Hard size limit: 6,000 lines per Rust file

**No Rust file may exceed 6,000 lines.** Before a file reaches that limit, split it
into a new module file and re-export what is needed.

- This includes `src/**/*.rs`, `tests/**/*.rs`, `benches/**/*.rs`, `examples/**/*.rs`.
- The limit is on physical lines in the file.
- 6,000 is the ceiling, not the target. Aim for **under 800 lines** per file.
- Split by responsibility, not by line count alone: one file = one job.
- If a `mod.rs` starts holding logic instead of just re-exports, split it.

Check before finishing work:

```bash
find src tests benches examples -name '*.rs' 2>/dev/null | xargs wc -l | sort -n
```

## 2. Modularity

- One module = one responsibility. No god modules.
- Public surface stays small: `pub` only what another module actually uses.
- No global mutable state. Pass state explicitly.
- Pure logic (parsing, harmony, key detection) must not depend on the CLI, files, or audio.
  That keeps it testable and reusable in WASM later.
- New behavior gets a unit test next to it. Test names say what is asserted, not "test1".

## 3. Language and dependencies

- **Rust only.** No Python, no virtualenv, no PyTorch/libtorch.
- Prefer pure-Rust crates. Avoid crates that bind C/C++ runtimes unless unavoidable.
- Pin dependencies in `Cargo.toml`; do not upgrade without a reason.
- No network calls at runtime, except the explicit `download` command.

## 4. Data and assets

- **Never download datasets automatically.** Ask the user first, then show the URL,
  size and license before running anything.
- Datasets go in `data/` and are gitignored.
- Do not commit audio (`.wav`), MIDI output, model checkpoints, or SoundFonts.

## 5. Explain, don't assume

- The user does not want to look up abbreviations. Expand every acronym on first use
  in docs and in CLI help (`GRU (gated recurrent unit)`).
- Prefer one small decision at a time over a large refactor.

## 6. Definition of done

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
```

All three must pass. No `#[allow(...)]` to silence a real warning.

## 7. Commit hygiene

- Small, focused commits with a message that says what changed and why.
- Never commit generated data, secrets, or `target/`.
