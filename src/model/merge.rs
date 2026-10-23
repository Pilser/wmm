//! Averaging several checkpoints into one.
//!
//! Why this exists: one training process uses roughly one CPU. The gated
//! recurrent unit (GRU) does 256 tiny matrix products per batch step, and
//! those are too small for Rayon (Candle's data-parallel executor) to split
//! usefully — measured speedup from 1 to 11 threads was about 7 %.
//!
//! So we parallelise *across processes* instead: start N workers from the same
//! initial weights, let each train on its own CPU (different data order), then
//! average the weights. This is local stochastic gradient descent (local SGD)
//! with one synchronisation point: model averaging.
//!
//! Averaging weights only makes sense when the workers started from the same
//! point, so every worker must be given the same `--init` checkpoint.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::Tensor;

use crate::model::gru::{build, count_parameters, ModelConfig};
use crate::model::train::{read_checkpoint_info, save_checkpoint, CheckpointInfo};

/// What the merge produced.
#[derive(Debug, Clone)]
pub struct MergeReport {
    pub inputs: usize,
    pub parameters: usize,
    pub steps: usize,
    pub model: ModelConfig,
}

/// Average the weights of all `inputs` and write the result to `out`.
pub fn merge(inputs: &[PathBuf], out: &Path) -> Result<MergeReport> {
    if inputs.is_empty() {
        bail!("merge needs at least one input checkpoint");
    }

    let mut infos = Vec::with_capacity(inputs.len());
    for path in inputs {
        infos.push(read_checkpoint_info(path)?);
    }

    let model = infos[0].model;
    for (path, info) in inputs.iter().zip(&infos) {
        if info.model != model {
            bail!(
                "{} has model shape {:?}, expected {:?}; cannot average different networks",
                path.display(),
                info.model,
                model
            );
        }
    }

    let mut maps = Vec::with_capacity(inputs.len());
    for path in inputs {
        let (_, mut varmap) = build(&model)?;
        varmap
            .load(path)
            .with_context(|| format!("cannot load {}", path.display()))?;
        maps.push(varmap);
    }

    let (_, out_map) = build(&model)?;
    {
        let out_data = out_map.data().lock().unwrap();
        for (name, var) in out_data.iter() {
            let mut tensors = Vec::with_capacity(maps.len());
            for map in &maps {
                let data = map.data().lock().unwrap();
                match data.get(name) {
                    Some(value) => tensors.push(value.as_tensor().clone()),
                    None => bail!("checkpoint is missing tensor {name}"),
                }
            }
            let references: Vec<&Tensor> = tensors.iter().collect();
            let mean = Tensor::stack(&references, 0)?.mean(0)?;
            var.set(&mean)?;
        }
    }

    let count = infos.len() as f64;
    let info = CheckpointInfo {
        model,
        steps: infos.iter().map(|info| info.steps).sum(),
        epochs: infos.iter().map(|info| info.epochs).max().unwrap_or(0),
        final_loss: infos.iter().map(|info| info.final_loss).sum::<f64>() / count,
        corpus_events: infos
            .iter()
            .map(|info| info.corpus_events)
            .max()
            .unwrap_or(0),
        seq_len: infos[0].seq_len,
    };
    save_checkpoint(&out_map, out, &info)?;

    Ok(MergeReport {
        inputs: inputs.len(),
        parameters: count_parameters(&out_map),
        steps: info.steps,
        model,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::train::CheckpointInfo;

    fn info(model: ModelConfig, steps: usize) -> CheckpointInfo {
        CheckpointInfo {
            model,
            steps,
            epochs: 1,
            final_loss: 1.0,
            corpus_events: 10,
            seq_len: 4,
        }
    }

    #[test]
    fn merged_weights_are_the_mean_of_the_inputs() {
        let dir = std::env::temp_dir().join(format!("wmm-merge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let model = ModelConfig {
            hidden: 8,
            embed: 4,
        };

        let (_, first) = build(&model).expect("build");
        let (_, second) = build(&model).expect("build");
        let a = dir.join("a.safetensors");
        let b = dir.join("b.safetensors");
        save_checkpoint(&first, &a, &info(model, 5)).expect("save a");
        save_checkpoint(&second, &b, &info(model, 7)).expect("save b");

        let out = dir.join("merged.safetensors");
        let report = merge(&[a.clone(), b.clone()], &out).expect("merge");
        assert_eq!(report.inputs, 2);
        assert_eq!(report.steps, 12);

        let (_, merged) = build(&model).expect("build");
        let mut merged = merged;
        merged.load(&out).expect("load merged");

        let a_data = first.data().lock().unwrap();
        let b_data = second.data().lock().unwrap();
        let m_data = merged.data().lock().unwrap();
        for (name, var) in a_data.iter() {
            let expected = ((var.as_tensor() + b_data[name].as_tensor()).unwrap() / 2.0).unwrap();
            let got = m_data[name].as_tensor();
            let diff = (expected - got).unwrap().abs().unwrap().sum_all().unwrap();
            let diff = diff.to_scalar::<f32>().unwrap();
            assert!(diff < 1e-5, "tensor {name} differs by {diff}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn merging_a_single_checkpoint_is_a_copy() {
        let dir = std::env::temp_dir().join(format!("wmm-copy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let model = ModelConfig {
            hidden: 4,
            embed: 2,
        };
        let (_, source) = build(&model).expect("build");
        let a = dir.join("a.safetensors");
        save_checkpoint(&source, &a, &info(model, 3)).expect("save");
        let out = dir.join("out.safetensors");
        merge(&[a], &out).expect("merge");

        let (_, mut copy) = build(&model).expect("build");
        copy.load(&out).expect("load");
        let source_data = source.data().lock().unwrap();
        let copy_data = copy.data().lock().unwrap();
        for (name, var) in source_data.iter() {
            let diff = (var.as_tensor() - copy_data[name].as_tensor())
                .unwrap()
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(diff < 1e-6, "tensor {name} differs by {diff}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
