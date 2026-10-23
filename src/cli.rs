//! Command-line interface. All user-facing behavior lives here; the library
//! modules stay free of argument parsing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

use crate::dataset;
use crate::demo;
use crate::events::{note_name, NoteStream};
use crate::harmonize::{self, Mode as HarmonyMode, Options as HarmonyOptions};
use crate::key::Key;
use crate::midi;
use crate::model::corpus::CorpusSplit;
use crate::model::vocab::stream_to_events;
use crate::model::{self, ModelConfig, SampleConfig, TrainConfig};
use crate::render::{self, RenderOptions};
use crate::stats::{self, Analysis, Totals};

/// System SoundFonts tried when `--sf2` is omitted.
const SOUNDFONT_CANDIDATES: &[&str] = &[
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/default.sf2",
];

#[derive(Debug, Parser)]
#[command(
    name = "wmm",
    version,
    about = "World Music Model — symbolic music generation (MIDI in, MIDI out)",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate the built-in demo phrase and its harmony.
    Demo(DemoArgs),
    /// Print note statistics for MIDI files or directories.
    Stats(StatsArgs),
    /// Add a rule-based harmony or echo to a MIDI file.
    Harmonize(HarmonizeArgs),
    /// Render a MIDI file to WAV with a SoundFont.
    Render(RenderArgs),
    /// Dump parsed note events, for debugging.
    Events(EventsArgs),
    /// Train the event GRU (gated recurrent unit) on MIDI files.
    Train(TrainArgs),
    /// Generate MIDI from a trained checkpoint.
    Sample(SampleArgs),
    /// Average several checkpoints into one (data-parallel training workers).
    Merge(MergeArgs),
}

#[derive(Debug, Args)]
struct DemoArgs {
    /// Output MIDI file.
    #[arg(short, long, default_value = "demo.mid")]
    out: PathBuf,
    /// Response type: unison, octave, third, fourth, fifth, sixth, or echo.
    #[arg(short = 'm', long, default_value = "third")]
    mode: String,
    /// Keep only the generated response, without the original phrase.
    #[arg(long)]
    response_only: bool,
    /// Also render to this WAV file.
    #[arg(long)]
    render: Option<PathBuf>,
    /// SoundFont (.sf2) for rendering. Defaults to a system GM SoundFont.
    #[arg(long)]
    sf2: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct StatsArgs {
    /// MIDI files or directories to scan.
    #[arg(required = true)]
    paths: Vec<PathBuf>,
    /// Dataset CSV index (e.g. maestro-v3.0.0.csv) for split labels and durations.
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Print machine-readable JSON instead of a table.
    #[arg(long)]
    json: bool,
    /// Maximum number of per-file rows to print (0 = all).
    #[arg(long, default_value_t = 20)]
    limit: usize,
}

#[derive(Debug, Args)]
struct HarmonizeArgs {
    /// Input MIDI file.
    #[arg(short, long)]
    r#in: PathBuf,
    /// Output MIDI file.
    #[arg(short, long)]
    out: PathBuf,
    /// Response type: unison, octave, third, fourth, fifth, sixth, or echo.
    #[arg(short = 'm', long, default_value = "third")]
    mode: String,
    /// Key such as `C`, `F# minor`, or `Am`. Detected from the input if omitted.
    #[arg(short = 'k', long)]
    key: Option<String>,
    /// Write only the response, not input + response.
    #[arg(long)]
    response_only: bool,
    /// MIDI channel for the response, 0–15.
    #[arg(long, default_value_t = 1)]
    channel: u8,
    /// Velocity multiplier for response notes, 0.0–1.0.
    #[arg(long, default_value_t = 0.85)]
    velocity_scale: f32,
    /// Echo delay in seconds (only for `--mode echo`).
    #[arg(long, default_value_t = 0.35)]
    echo_delay: f64,
}

#[derive(Debug, Args)]
struct RenderArgs {
    /// Input MIDI file.
    #[arg(short, long)]
    r#in: PathBuf,
    /// Output WAV file.
    #[arg(short, long)]
    out: PathBuf,
    /// SoundFont (.sf2). Defaults to a system GM SoundFont.
    #[arg(long)]
    sf2: Option<PathBuf>,
    /// Sample rate in hertz.
    #[arg(long, default_value_t = 44_100)]
    rate: u32,
    /// Output gain, 1.0 = unchanged.
    #[arg(long, default_value_t = 1.0)]
    gain: f32,
}

#[derive(Debug, Args)]
struct EventsArgs {
    /// Input MIDI file.
    #[arg(short, long)]
    r#in: PathBuf,
    /// Print at most this many notes (0 = all).
    #[arg(long, default_value_t = 40)]
    limit: usize,
}

#[derive(Debug, Args)]
struct TrainArgs {
    /// MIDI file or directory with training data.
    #[arg(long, default_value = "data/maestro")]
    data: PathBuf,
    /// Use at most this many files (0 = all). Keep the smoke test small.
    #[arg(long, default_value_t = 10)]
    files: usize,
    /// Output checkpoint (`.safetensors`). A `.json` sidecar stores the shape.
    #[arg(short, long, default_value = "runs/smoke.safetensors")]
    out: PathBuf,
    /// Number of passes over the data.
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    /// Windows per optimizer step.
    #[arg(long, default_value_t = 8)]
    batch: usize,
    /// Events per training window.
    #[arg(long, default_value_t = 256)]
    seq: usize,
    /// GRU hidden size.
    #[arg(long, default_value_t = 128)]
    hidden: usize,
    /// Size of each input embedding.
    #[arg(long, default_value_t = 64)]
    embed: usize,
    /// AdamW learning rate.
    #[arg(long, default_value_t = 3e-3)]
    lr: f64,
    /// Stop after this many optimizer steps (0 = no limit).
    #[arg(long, default_value_t = 0)]
    max_steps: usize,
    /// Stop after this many minutes (0 = no limit).
    #[arg(long, default_value_t = 0.0)]
    max_minutes: f64,
    /// Print progress every N steps.
    #[arg(long, default_value_t = 10)]
    log_every: usize,
    /// Write an intermediate checkpoint every N steps (0 = only at the end).
    #[arg(long, default_value_t = 0)]
    save_every: usize,
    /// Start from these weights (used by data-parallel workers so that their
    /// weights can be averaged afterwards).
    #[arg(long)]
    init: Option<PathBuf>,
    /// Only train on this manifest split (MAESTRO CSV index): train,
    /// validation, test, or all. Default: all.
    #[arg(long, default_value = "all")]
    split: String,
    /// Cap on windows used for the final train/validation report (0 = all).
    #[arg(long, default_value_t = 512)]
    eval_windows: usize,
    /// Random seed for shuffling and sampling.
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Debug, Args)]
struct SampleArgs {
    /// Checkpoint written by `wmm train`.
    #[arg(short, long)]
    checkpoint: PathBuf,
    /// Output MIDI file.
    #[arg(short, long, default_value = "gen.mid")]
    out: PathBuf,
    /// Maximum number of generated events.
    #[arg(long, default_value_t = 400)]
    events: usize,
    /// Softmax temperature: lower is safer, higher is wilder.
    #[arg(long, default_value_t = 0.9)]
    temperature: f32,
    /// Stop after this many seconds of music.
    #[arg(long, default_value_t = 120.0)]
    max_seconds: f64,
    /// Prime the model with the first events of this MIDI file.
    #[arg(long)]
    prompt: Option<PathBuf>,
    /// Also render the result to this WAV file.
    #[arg(long)]
    render: Option<PathBuf>,
    /// SoundFont (.sf2) for rendering.
    #[arg(long)]
    sf2: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

/// Parse arguments and execute the requested command.
pub fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Demo(args) => run_demo(args),
        Command::Stats(args) => run_stats(args),
        Command::Harmonize(args) => run_harmonize(args),
        Command::Render(args) => run_render(args),
        Command::Events(args) => run_events(args),
        Command::Train(args) => run_train(args),
        Command::Sample(args) => run_sample(args),
        Command::Merge(args) => run_merge(args),
    }
}

fn run_demo(args: DemoArgs) -> Result<()> {
    let phrase = demo::phrase();
    // The demo is composed in A minor; detecting it would be ambiguous with the
    // relative major, so we use the known key.
    let key = demo::key();
    let mode = HarmonyMode::parse(&args.mode)?;
    let options = HarmonyOptions {
        mode,
        key: Some(key),
        velocity_scale: 0.8,
        echo_delay: 0.35,
    };
    let response = harmonize::assign_channel(&harmonize::harmonize(&phrase, &options), 1);

    let output = if args.response_only {
        response
    } else {
        let mut merged = phrase.clone();
        merged.extend(&response);
        merged
    };

    midi::save(&args.out, &output)?;
    let analysis = stats::analyze(&output);
    println!("wrote {}", args.out.display());
    println!(
        "  {} notes, {:.2} s, range {}, key {}",
        analysis.notes,
        analysis.duration_s,
        analysis.pitch_range(),
        key.name()
    );
    println!("  response: {} on channel 1", mode);

    if let Some(wav) = args.render {
        let sf2 = resolve_soundfont(args.sf2)?;
        let report = render::render_midi_to_wav(&args.out, &sf2, &wav, &RenderOptions::default())?;
        println!(
            "wrote {} ({:.2} s at {} Hz)",
            wav.display(),
            report.seconds,
            report.sample_rate
        );
    }
    Ok(())
}

fn run_stats(args: StatsArgs) -> Result<()> {
    let mut files = Vec::new();
    for path in &args.paths {
        files.extend(dataset::find_midi_files(path)?);
    }
    if files.is_empty() {
        println!("no MIDI files found");
        return Ok(());
    }

    let manifest = match args.manifest {
        Some(path) => Some(dataset::Manifest::load(&path)?),
        None => auto_manifest(&args.paths),
    };

    let mut rows: Vec<(PathBuf, Analysis, f64, dataset::Split)> = Vec::with_capacity(files.len());
    let mut totals = Totals::default();
    let mut by_split: BTreeMap<dataset::Split, Totals> = BTreeMap::new();
    let mut failures = 0usize;

    for path in &files {
        match midi::load(path) {
            Ok(loaded) => {
                let analysis = stats::analyze(&loaded.stream);
                let bpm = loaded.tempo.bpm_at(0);
                let split = manifest
                    .as_ref()
                    .map(|m| m.split_for(path))
                    .unwrap_or_else(|| dataset::split_of(path));
                totals.add(&analysis);
                by_split.entry(split).or_default().add(&analysis);
                rows.push((path.clone(), analysis, bpm, split));
            }
            Err(error) => {
                failures += 1;
                eprintln!("skip {}: {error}", path.display());
            }
        }
    }

    if args.json {
        print_stats_json(&rows, &totals, &by_split, manifest.as_ref(), failures);
        return Ok(());
    }

    let shown = if args.limit == 0 {
        rows.len()
    } else {
        rows.len().min(args.limit)
    };
    println!(
        "{:<40} {:>6} {:>7} {:>8} {:>10} {:>5} {:>7}",
        "file", "split", "notes", "seconds", "range", "vel", "poly"
    );
    for (path, analysis, _, split) in rows.iter().take(shown) {
        println!(
            "{:<40} {:>6} {:>7} {:>8.1} {:>10} {:>5.0} {:>7.2}",
            truncate(&path.display().to_string(), 40),
            split.as_str(),
            analysis.notes,
            analysis.duration_s,
            analysis.pitch_range(),
            analysis.mean_velocity,
            analysis.mean_polyphony
        );
    }
    if shown < rows.len() {
        println!("... {} more files", rows.len() - shown);
    }
    if failures > 0 {
        println!("{} files failed to parse", failures);
    }

    println!();
    print_totals("all", &totals);
    if by_split.len() > 1 {
        println!();
        for (split, split_totals) in &by_split {
            print_totals(split.as_str(), split_totals);
        }
    }
    if let Some(manifest) = &manifest {
        println!();
        println!(
            "CSV declares   {:.2} h across {} files; parsed notes span {:.2} h",
            manifest.declared_duration() / 3600.0,
            manifest.len(),
            totals.duration_s / 3600.0
        );
    }
    Ok(())
}

fn print_totals(label: &str, totals: &Totals) {
    println!(
        "{label:<12} {:>5} files {:>9} notes {:>9.2} h  range {:<9} vel {:>5.1} poly {:.2}",
        totals.files,
        totals.notes,
        totals.duration_s / 3600.0,
        totals.pitch_range(),
        totals.mean_velocity(),
        totals.mean_polyphony()
    );
}

fn auto_manifest(paths: &[PathBuf]) -> Option<dataset::Manifest> {
    for path in paths {
        let dir = if path.is_dir() {
            path.clone()
        } else {
            path.parent()?.to_path_buf()
        };
        if let Some(csv) = dataset::find_manifest(&dir) {
            if let Ok(manifest) = dataset::Manifest::load(&csv) {
                return Some(manifest);
            }
        }
    }
    None
}

fn print_stats_json(
    rows: &[(PathBuf, Analysis, f64, dataset::Split)],
    totals: &Totals,
    by_split: &BTreeMap<dataset::Split, Totals>,
    manifest: Option<&dataset::Manifest>,
    failures: usize,
) {
    let files: Vec<serde_json::Value> = rows
        .iter()
        .map(|(path, analysis, bpm, split)| {
            serde_json::json!({
                "path": path.display().to_string(),
                "split": split.as_str(),
                "notes": analysis.notes,
                "seconds": analysis.duration_s,
                "min_pitch": analysis.min_pitch,
                "max_pitch": analysis.max_pitch,
                "mean_velocity": analysis.mean_velocity,
                "mean_polyphony": analysis.mean_polyphony,
                "density": analysis.density(),
                "tempo_bpm": bpm,
            })
        })
        .collect();
    let splits: serde_json::Value = by_split
        .iter()
        .map(|(split, totals)| {
            (
                split.as_str().to_string(),
                serde_json::json!({
                    "files": totals.files,
                    "notes": totals.notes,
                    "hours": totals.duration_s / 3600.0,
                    "mean_velocity": totals.mean_velocity(),
                    "mean_polyphony": totals.mean_polyphony(),
                }),
            )
        })
        .collect::<serde_json::Map<String, serde_json::Value>>()
        .into();
    let value = serde_json::json!({
        "files": files,
        "failures": failures,
        "splits": splits,
        "manifest": manifest.map(|m| serde_json::json!({
            "files": m.len(),
            "declared_hours": m.declared_duration() / 3600.0,
        })),
        "totals": {
            "files": totals.files,
            "notes": totals.notes,
            "seconds": totals.duration_s,
            "min_pitch": totals.min_pitch,
            "max_pitch": totals.max_pitch,
            "mean_velocity": totals.mean_velocity(),
            "mean_polyphony": totals.mean_polyphony(),
            "density": totals.density(),
        }
    });
    match serde_json::to_string_pretty(&value) {
        Ok(text) => println!("{text}"),
        Err(error) => eprintln!("cannot serialize stats: {error}"),
    }
}

fn run_harmonize(args: HarmonizeArgs) -> Result<()> {
    let loaded = midi::load(&args.r#in)?;
    if loaded.stream.is_empty() {
        anyhow::bail!("input contains no notes: {}", args.r#in.display());
    }

    let key = match &args.key {
        Some(text) => Key::parse(text)?,
        None => Key::detect(&loaded.stream.notes),
    };
    let mode = HarmonyMode::parse(&args.mode)?;
    let options = HarmonyOptions {
        mode,
        key: Some(key),
        velocity_scale: args.velocity_scale,
        echo_delay: args.echo_delay,
    };
    let response = harmonize::assign_channel(
        &harmonize::harmonize(&loaded.stream, &options),
        args.channel,
    );

    let output = if args.response_only {
        response
    } else {
        let mut merged = loaded.stream.clone();
        merged.extend(&response);
        merged
    };

    midi::save(&args.out, &output)?;
    println!(
        "wrote {} ({} notes, mode {}, key {}, response on channel {})",
        args.out.display(),
        output.len(),
        mode,
        key.name(),
        args.channel
    );
    Ok(())
}

fn run_render(args: RenderArgs) -> Result<()> {
    let sf2 = resolve_soundfont(args.sf2)?;
    let options = RenderOptions {
        sample_rate: args.rate,
        gain: args.gain,
        ..RenderOptions::default()
    };
    let report = render::render_midi_to_wav(&args.r#in, &sf2, &args.out, &options)?;
    println!(
        "wrote {} ({:.2} s, {} frames at {} Hz)",
        args.out.display(),
        report.seconds,
        report.frames,
        report.sample_rate
    );
    Ok(())
}

fn run_events(args: EventsArgs) -> Result<()> {
    let loaded = midi::load(&args.r#in)?;
    println!(
        "{}: {} notes, {} tracks, tempo {:.1} BPM",
        loaded.name.as_deref().unwrap_or("(unnamed)"),
        loaded.stream.len(),
        loaded.track_count,
        loaded.tempo.bpm_at(0)
    );
    println!(
        "{:>8} {:>7} {:>6} {:>4} {:>3}",
        "start", "dur", "pitch", "vel", "ch"
    );
    let shown = if args.limit == 0 {
        loaded.stream.len()
    } else {
        loaded.stream.len().min(args.limit)
    };
    for note in loaded.stream.notes.iter().take(shown) {
        println!(
            "{:>8.3} {:>7.3} {:>6} {:>4} {:>3}",
            note.start,
            note.duration,
            note_name(note.pitch),
            note.velocity,
            note.channel
        );
    }
    if shown < loaded.stream.len() {
        println!("... {} more notes", loaded.stream.len() - shown);
    }
    Ok(())
}

fn run_train(args: TrainArgs) -> Result<()> {
    let config = TrainConfig {
        epochs: args.epochs,
        batch_size: args.batch,
        seq_len: args.seq,
        learning_rate: args.lr,
        model: ModelConfig {
            hidden: args.hidden,
            embed: args.embed,
        },
        max_steps: args.max_steps,
        max_minutes: args.max_minutes,
        log_every: args.log_every,
        save_every: args.save_every,
        init: args.init,
        split: parse_split(&args.split)?,
        eval_windows: args.eval_windows,
        seed: args.seed,
        ..TrainConfig::default()
    };
    let file_limit = if args.files == 0 {
        None
    } else {
        Some(args.files)
    };

    let report = model::run(&args.data, file_limit, &config, &args.out)?;
    let events_per_second = if report.seconds > 0.0 {
        (report.steps * config.batch_size * config.seq_len) as f64 / report.seconds
    } else {
        0.0
    };
    println!();
    println!("checkpoint      {}", args.out.display());
    println!("parameters      {}", report.parameters);
    println!(
        "data            {} files, {} events, {} windows ({} train / {} val)",
        report.files,
        report.corpus_events,
        report.windows,
        report.train_windows,
        report.val_windows
    );
    println!(
        "loss            {:.4} → {:.4} (start → last), train {:.4}, val {:.4}",
        report.initial_loss, report.final_loss, report.train_loss, report.val_loss
    );
    println!(
        "steps           {} in {:.1} s ({:.0} events/s)",
        report.steps, report.seconds, events_per_second
    );
    println!(
        "next            wmm sample --checkpoint {} --out gen.mid",
        args.out.display()
    );
    Ok(())
}

fn run_sample(args: SampleArgs) -> Result<()> {
    let (model, _varmap, info) = model::load_checkpoint(&args.checkpoint)?;
    let prompt = match &args.prompt {
        Some(path) => {
            let loaded = midi::load(path)?;
            let events = stream_to_events(&loaded.stream);
            events.into_iter().take(128).collect()
        }
        None => Vec::new(),
    };

    let config = SampleConfig {
        events: args.events,
        temperature: args.temperature,
        seed: args.seed,
        max_seconds: args.max_seconds,
        tail: 0.5,
    };
    let stream = model::generate(&model, &prompt, &config)?;
    midi::save(&args.out, &stream)?;
    println!(
        "wrote {} ({} notes, {:.1} s) from a checkpoint trained for {} steps",
        args.out.display(),
        stream.len(),
        stream.duration(),
        info.steps
    );

    if let Some(wav) = args.render {
        let sf2 = resolve_soundfont(args.sf2)?;
        let report = render::render_midi_to_wav(&args.out, &sf2, &wav, &RenderOptions::default())?;
        println!(
            "wrote {} ({:.2} s at {} Hz)",
            wav.display(),
            report.seconds,
            report.sample_rate
        );
    }
    Ok(())
}

#[derive(Debug, Args)]
struct MergeArgs {
    /// Checkpoints to average, given in any order.
    #[arg(long, num_args = 1.., required = true)]
    inputs: Vec<PathBuf>,
    /// Output checkpoint.
    #[arg(short, long)]
    out: PathBuf,
}

fn run_merge(args: MergeArgs) -> Result<()> {
    let report = model::merge(&args.inputs, &args.out)?;
    println!(
        "merged {} checkpoints ({} parameters, {} recorded steps) into {}",
        report.inputs,
        report.parameters,
        report.steps,
        args.out.display()
    );
    Ok(())
}

/// Parse the `--split` CLI value.
fn parse_split(text: &str) -> Result<CorpusSplit> {
    match text.trim().to_lowercase().as_str() {
        "train" => Ok(CorpusSplit::Train),
        "validation" | "val" => Ok(CorpusSplit::Validation),
        "test" => Ok(CorpusSplit::Test),
        "all" => Ok(CorpusSplit::All),
        other => anyhow::bail!("unknown split {other:?}; expected train, validation, test, or all"),
    }
}

fn resolve_soundfont(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if !path.exists() {
            anyhow::bail!("SoundFont not found: {}", path.display());
        }
        return Ok(path);
    }
    for candidate in SOUNDFONT_CANDIDATES {
        let path = Path::new(candidate);
        if path.exists() {
            return Ok(path.to_path_buf());
        }
    }
    Err(anyhow::anyhow!(
        "no system SoundFont found; pass --sf2 /path/to/piano.sf2 (see docs/01-piano-poc.md)"
    ))
}

/// Shorten text to `max` characters, adding an ellipsis.
fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Convenience for tests: parse a note stream without touching the CLI.
#[allow(dead_code)]
pub(crate) fn notes_of(stream: &NoteStream) -> Vec<String> {
    stream.notes.iter().map(|n| n.name()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_respects_char_boundaries() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("abcdefgh", 5), "abcd…");
    }

    #[test]
    fn soundfont_resolution_reports_missing_file() {
        let error = resolve_soundfont(Some(PathBuf::from("/nonexistent/file.sf2"))).unwrap_err();
        assert!(error.to_string().contains("SoundFont not found"));
    }

    #[test]
    fn helpers_compile() {
        let stream = NoteStream::new();
        assert!(notes_of(&stream).is_empty());
    }
}
