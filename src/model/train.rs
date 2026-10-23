//! Training loop for the event GRU.
//!
//! Objective: given the previous events, predict the next one. Four
//! cross-entropy losses (time, kind, pitch, velocity) are averaged, so each
//! head contributes equally regardless of its number of classes.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::loss::cross_entropy;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::{VarBuilder, VarMap};

use crate::model::corpus::{Corpus, CorpusSplit};
use crate::model::gru::{count_parameters, EventModel, ModelConfig};
use crate::model::rng::Rng;
use crate::model::vocab::{Event, KIND_COUNT, KIND_START, PITCH_COUNT, TIME_BINS, VEL_BINS};

/// Everything that controls a training run.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    /// Step between windows. 0 means "no overlap" (stride = seq_len).
    pub stride: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub model: ModelConfig,
    /// Stop after this many optimizer steps (0 = no limit).
    pub max_steps: usize,
    /// Stop after this many minutes (0 = no limit).
    pub max_minutes: f64,
    pub log_every: usize,
    /// Fraction of windows held out for validation.
    pub val_fraction: f64,
    pub seed: u64,
    /// Start from these weights instead of a random initialisation.
    pub init: Option<std::path::PathBuf>,
    /// Restrict training to a manifest split (MAESTRO CSV index).
    pub split: CorpusSplit,
    /// Cap on windows used by the final quality report (0 = all).
    ///
    /// Without this, a multi-hour run pays for a second full pass at the end.
    pub eval_windows: usize,
    /// Write an intermediate checkpoint every N steps (0 = only at the end).
    pub save_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            batch_size: 8,
            seq_len: 256,
            stride: 0,
            learning_rate: 3e-3,
            weight_decay: 0.01,
            model: ModelConfig::default(),
            max_steps: 0,
            max_minutes: 0.0,
            log_every: 10,
            val_fraction: 0.05,
            seed: 1,
            init: None,
            split: CorpusSplit::All,
            eval_windows: 512,
            save_every: 0,
        }
    }
}

/// What happened, for the CLI summary.
#[derive(Debug, Clone)]
pub struct TrainReport {
    pub files: usize,
    pub corpus_events: usize,
    pub windows: usize,
    pub train_windows: usize,
    pub val_windows: usize,
    pub parameters: usize,
    pub steps: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub train_loss: f64,
    pub val_loss: f64,
    pub seconds: f64,
}

/// Shape saved next to the weights.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointInfo {
    pub model: ModelConfig,
    pub steps: usize,
    pub epochs: usize,
    pub final_loss: f64,
    pub corpus_events: usize,
    pub seq_len: usize,
}

struct Batch {
    in_time: Tensor,
    in_kind: Tensor,
    in_pitch: Tensor,
    in_velocity: Tensor,
    tg_time: Tensor,
    tg_kind: Tensor,
    tg_pitch: Tensor,
    tg_velocity: Tensor,
    size: usize,
}

/// Train on a MIDI file or directory and write a checkpoint.
pub fn run(
    data: &Path,
    file_limit: Option<usize>,
    config: &TrainConfig,
    out: &Path,
) -> Result<TrainReport> {
    let started = Instant::now();
    let corpus = Corpus::load_split(data, file_limit, config.split)?;
    let stride = if config.stride == 0 {
        config.seq_len
    } else {
        config.stride
    };
    let mut windows = corpus.windows(config.seq_len, stride);
    if windows.is_empty() {
        anyhow::bail!(
            "not enough events ({} in {} files) for windows of {}; use more files or a smaller --seq",
            corpus.events,
            corpus.files,
            config.seq_len
        );
    }

    let mut rng = Rng::new(config.seed);
    rng.shuffle(&mut windows);
    let val_count = ((windows.len() as f64 * config.val_fraction).round() as usize)
        .min(windows.len().saturating_sub(1));
    let (val, train) = windows.split_at(val_count);
    let train_windows = train.len();
    let val_windows = val.len();

    let device = Device::Cpu;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = EventModel::new(vb, &config.model)?;
    if let Some(init) = &config.init {
        varmap
            .load(init)
            .with_context(|| format!("cannot initialise from {}", init.display()))?;
        println!("initialised weights from {}", init.display());
    }
    let parameters = count_parameters(&varmap);

    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..Default::default()
        },
    )?;

    println!(
        "corpus: {} files, {} events, {} windows ({} train / {} val)",
        corpus.files,
        corpus.events,
        windows.len(),
        train_windows,
        val_windows
    );
    println!(
        "model: hidden {}, embed {}, {} parameters",
        config.model.hidden, config.model.embed, parameters
    );

    let mut initial_loss = f64::NAN;
    let mut latest_loss = f64::NAN;
    let mut steps = 0usize;
    let mut epochs_done = 0usize;
    let limit_minutes = if config.max_minutes > 0.0 {
        Some(config.max_minutes)
    } else {
        None
    };

    'epochs: for epoch in 0..config.epochs {
        let mut order: Vec<usize> = (0..train_windows).collect();
        rng.shuffle(&mut order);

        for chunk in order.chunks(config.batch_size) {
            let batch = make_batch(train, chunk, config.seq_len, &device)?;
            let loss = forward_loss(&model, &batch, config.seq_len)?;
            let value = loss.to_scalar::<f32>()? as f64;
            optimizer.backward_step(&loss)?;

            if initial_loss.is_nan() {
                initial_loss = value;
            }
            latest_loss = value;
            steps += 1;

            if config.log_every > 0 && steps % config.log_every == 0 {
                let elapsed = started.elapsed().as_secs_f64();
                let rate = (steps * config.batch_size * config.seq_len) as f64 / elapsed;
                println!(
                    "epoch {epoch} step {steps}: loss {value:.4}  ({rate:.0} events/s, {elapsed:.1}s)"
                );
            }

            if config.save_every > 0 && steps % config.save_every == 0 {
                save_checkpoint(
                    &varmap,
                    out,
                    &CheckpointInfo {
                        model: config.model,
                        steps,
                        epochs: epoch + 1,
                        final_loss: latest_loss,
                        corpus_events: corpus.events,
                        seq_len: config.seq_len,
                    },
                )?;
                println!("saved checkpoint after {steps} steps");
            }

            if config.max_steps > 0 && steps >= config.max_steps {
                epochs_done = epoch + 1;
                break 'epochs;
            }
            if let Some(limit) = limit_minutes {
                if started.elapsed().as_secs_f64() >= limit * 60.0 {
                    println!("time limit reached ({limit} min)");
                    epochs_done = epoch + 1;
                    break 'epochs;
                }
            }
        }
        epochs_done = epoch + 1;

        if !val.is_empty() {
            let value = evaluate(&model, val, config, config.eval_windows)?;
            println!("epoch {epoch} validation loss {value:.4}");
        }
    }

    let info = CheckpointInfo {
        model: config.model,
        steps,
        epochs: epochs_done,
        final_loss: latest_loss,
        corpus_events: corpus.events,
        seq_len: config.seq_len,
    };
    save_checkpoint(&varmap, out, &info)?;

    let train_loss = evaluate(&model, train, config, config.eval_windows)?;
    let val_loss = if val.is_empty() {
        f64::NAN
    } else {
        evaluate(&model, val, config, config.eval_windows)?
    };

    Ok(TrainReport {
        files: corpus.files,
        corpus_events: corpus.events,
        windows: windows.len(),
        train_windows,
        val_windows,
        parameters,
        steps,
        initial_loss,
        final_loss: latest_loss,
        train_loss,
        val_loss,
        seconds: started.elapsed().as_secs_f64(),
    })
}

fn forward_loss(model: &EventModel, batch: &Batch, seq_len: usize) -> Result<Tensor> {
    let logits = model.forward(
        &batch.in_time,
        &batch.in_kind,
        &batch.in_pitch,
        &batch.in_velocity,
    )?;
    let [time, kind, pitch, velocity] = logits.flattened()?;
    let n = batch.size * seq_len;
    let l_time = cross_entropy(&time, &batch.tg_time.reshape(n)?)?;
    let l_kind = cross_entropy(&kind, &batch.tg_kind.reshape(n)?)?;
    let l_pitch = cross_entropy(&pitch, &batch.tg_pitch.reshape(n)?)?;
    let l_velocity = cross_entropy(&velocity, &batch.tg_velocity.reshape(n)?)?;
    let total = ((l_time + l_kind)? + (l_pitch + l_velocity)?)?;
    Ok((total / 4.0)?)
}

/// Mean loss over up to `max_windows` windows, without gradients.
fn evaluate(
    model: &EventModel,
    windows: &[Vec<Event>],
    config: &TrainConfig,
    max_windows: usize,
) -> Result<f64> {
    if windows.is_empty() {
        return Ok(f64::NAN);
    }
    let take = if max_windows == 0 {
        windows.len()
    } else {
        windows.len().min(max_windows)
    };
    let device = Device::Cpu;
    let mut total = 0.0;
    let mut batches = 0.0;
    for chunk in windows[..take].chunks(config.batch_size.max(1)) {
        let ids: Vec<usize> = (0..chunk.len()).collect();
        let batch = make_batch(windows, &ids, config.seq_len, &device)?;
        let loss = forward_loss(model, &batch, config.seq_len)?;
        total += loss.to_scalar::<f32>()? as f64;
        batches += 1.0;
    }
    Ok(total / batches)
}

/// Write weights plus the JSON sidecar that records their shape.
pub(crate) fn save_checkpoint(varmap: &VarMap, out: &Path, info: &CheckpointInfo) -> Result<()> {
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("cannot create {}", parent.display()))?;
        }
    }
    varmap
        .save(out)
        .with_context(|| format!("cannot save checkpoint {}", out.display()))?;
    let info_path = out.with_extension("json");
    std::fs::write(&info_path, serde_json::to_vec_pretty(info)?)
        .with_context(|| format!("cannot write {}", info_path.display()))?;
    Ok(())
}

/// Build tensors for a batch. Each window has `seq_len + 1` events: the first
/// `seq_len` are inputs and the shifted version is the target.
fn make_batch(
    windows: &[Vec<Event>],
    ids: &[usize],
    seq_len: usize,
    device: &Device,
) -> Result<Batch> {
    let size = ids.len();
    let mut in_time = Vec::with_capacity(size * seq_len);
    let mut in_kind = Vec::with_capacity(size * seq_len);
    let mut in_pitch = Vec::with_capacity(size * seq_len);
    let mut in_velocity = Vec::with_capacity(size * seq_len);
    let mut tg_time = Vec::with_capacity(size * seq_len);
    let mut tg_kind = Vec::with_capacity(size * seq_len);
    let mut tg_pitch = Vec::with_capacity(size * seq_len);
    let mut tg_velocity = Vec::with_capacity(size * seq_len);

    for &id in ids {
        let window = &windows[id];
        let usable = window.len().min(seq_len + 1);
        for step in 0..seq_len {
            let input = window.get(step).copied().unwrap_or_else(Event::start);
            let target = window.get(step + 1).copied().unwrap_or_else(Event::start);
            let _ = usable;
            in_time.push(input.dt_bin());
            in_kind.push(input.kind);
            in_pitch.push(input.pitch as u32);
            in_velocity.push(input.vel_bin());
            tg_time.push(target.dt_bin());
            tg_kind.push(target.kind);
            tg_pitch.push(target.pitch as u32);
            tg_velocity.push(target.vel_bin());
        }
    }

    let shape = (size, seq_len);
    Ok(Batch {
        in_time: Tensor::from_vec(in_time, shape, device)?,
        in_kind: Tensor::from_vec(in_kind, shape, device)?,
        in_pitch: Tensor::from_vec(in_pitch, shape, device)?,
        in_velocity: Tensor::from_vec(in_velocity, shape, device)?,
        tg_time: Tensor::from_vec(tg_time, shape, device)?,
        tg_kind: Tensor::from_vec(tg_kind, shape, device)?,
        tg_pitch: Tensor::from_vec(tg_pitch, shape, device)?,
        tg_velocity: Tensor::from_vec(tg_velocity, shape, device)?,
        size,
    })
}

/// Load a checkpoint and the shapes recorded next to it.
pub fn load_checkpoint(path: &Path) -> Result<(EventModel, VarMap, CheckpointInfo)> {
    let info = read_checkpoint_info(path)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let model = EventModel::new(vb, &info.model)?;
    let mut varmap = varmap;
    varmap
        .load(path)
        .with_context(|| format!("cannot load weights {}", path.display()))?;
    Ok((model, varmap, info))
}

/// Read only the JSON sidecar (shapes and run metadata).
pub fn read_checkpoint_info(path: &Path) -> Result<CheckpointInfo> {
    let info_path = path.with_extension("json");
    let text = std::fs::read_to_string(&info_path)
        .with_context(|| format!("cannot read {}", info_path.display()))?;
    serde_json::from_str(&text).context("checkpoint metadata is not valid JSON")
}

/// Class counts, used by the trainer and sampler to sanity-check the vocabulary.
pub fn vocab_sizes() -> [usize; 4] {
    [TIME_BINS, KIND_COUNT, PITCH_COUNT, VEL_BINS]
}

/// The token that begins a generated sequence.
pub fn start_token() -> Event {
    Event {
        kind: KIND_START,
        ..Event::start()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_sizes_match_the_heads() {
        assert_eq!(vocab_sizes(), [128, 3, 128, 32]);
    }

    #[test]
    fn batch_shape_is_batch_by_sequence() {
        let windows = vec![vec![Event::start(); 5], vec![Event::start(); 5]];
        let batch = make_batch(&windows, &[0, 1], 4, &Device::Cpu).expect("batch");
        assert_eq!(batch.in_time.dims(), &[2, 4]);
        assert_eq!(batch.tg_pitch.dims(), &[2, 4]);
    }
}
