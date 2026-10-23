//! A small gated recurrent unit (GRU) over the event vocabulary.
//!
//! One layer is enough for the first version: the model has to learn local
//! harmony and rhythm, not long-range structure. The planner model (phrase
//! rate) comes later and will sit above this one.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::rnn::{gru, GRUConfig, GRUState, RNN};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};

use crate::model::vocab::{KIND_COUNT, PITCH_COUNT, TIME_BINS, VEL_BINS};

/// Shape of the network, saved next to the weights so sampling can rebuild it.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    /// Size of the GRU hidden state.
    pub hidden: usize,
    /// Size of each of the four input embeddings.
    pub embed: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden: 128,
            embed: 64,
        }
    }
}

impl ModelConfig {
    fn input_size(&self) -> usize {
        self.embed * 4
    }
}

/// Output distributions for one or more steps. Each tensor is `[batch, time, classes]`.
#[derive(Debug)]
pub struct Logits {
    pub time: Tensor,
    pub kind: Tensor,
    pub pitch: Tensor,
    pub velocity: Tensor,
}

impl Logits {
    /// Flatten `[b, t, c]` to `[b*t, c]` for cross-entropy.
    pub fn flattened(&self) -> Result<[Tensor; 4]> {
        let (b, t, _) = self.time.dims3()?;
        let n = b * t;
        Ok([
            self.time.reshape((n, TIME_BINS))?,
            self.kind.reshape((n, KIND_COUNT))?,
            self.pitch.reshape((n, PITCH_COUNT))?,
            self.velocity.reshape((n, VEL_BINS))?,
        ])
    }
}

/// The event model: four embeddings, one GRU, four prediction heads.
pub struct EventModel {
    gru: candle_nn::rnn::GRU,
    emb_time: Embedding,
    emb_kind: Embedding,
    emb_pitch: Embedding,
    emb_velocity: Embedding,
    head_time: Linear,
    head_kind: Linear,
    head_pitch: Linear,
    head_velocity: Linear,
    config: ModelConfig,
}

impl EventModel {
    pub fn new(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let hidden = config.hidden;
        Ok(Self {
            gru: gru(
                config.input_size(),
                hidden,
                GRUConfig::default(),
                vb.pp("gru"),
            )?,
            emb_time: embedding(TIME_BINS, config.embed, vb.pp("emb_time"))?,
            emb_kind: embedding(KIND_COUNT, config.embed, vb.pp("emb_kind"))?,
            emb_pitch: embedding(PITCH_COUNT, config.embed, vb.pp("emb_pitch"))?,
            emb_velocity: embedding(VEL_BINS, config.embed, vb.pp("emb_velocity"))?,
            head_time: linear(hidden, TIME_BINS, vb.pp("head_time"))?,
            head_kind: linear(hidden, KIND_COUNT, vb.pp("head_kind"))?,
            head_pitch: linear(hidden, PITCH_COUNT, vb.pp("head_pitch"))?,
            head_velocity: linear(hidden, VEL_BINS, vb.pp("head_velocity"))?,
            config: *config,
        })
    }

    pub fn config(&self) -> ModelConfig {
        self.config
    }

    pub fn zero_state(&self, batch: usize) -> Result<GRUState> {
        Ok(self.gru.zero_state(batch)?)
    }

    /// Embed four `[batch, time]` integer tensors into `[batch, time, 4*embed]`.
    fn embed(
        &self,
        time: &Tensor,
        kind: &Tensor,
        pitch: &Tensor,
        velocity: &Tensor,
    ) -> Result<Tensor> {
        let e_time = self.emb_time.forward(time)?;
        let e_kind = self.emb_kind.forward(kind)?;
        let e_pitch = self.emb_pitch.forward(pitch)?;
        let e_velocity = self.emb_velocity.forward(velocity)?;
        Ok(Tensor::cat(&[&e_time, &e_kind, &e_pitch, &e_velocity], 2)?)
    }

    /// Run a whole sequence, resetting the recurrent state at the start.
    pub fn forward(
        &self,
        time: &Tensor,
        kind: &Tensor,
        pitch: &Tensor,
        velocity: &Tensor,
    ) -> Result<Logits> {
        let embedded = self.embed(time, kind, pitch, velocity)?;
        let states = self.gru.seq(&embedded)?;
        let hidden: Vec<Tensor> = states.iter().map(|state| state.h().clone()).collect();
        let stacked = Tensor::stack(&hidden, 1)?;
        self.heads(&stacked)
    }

    /// One autoregressive step. Used by the sampler.
    pub fn step(
        &self,
        time: u32,
        kind: u32,
        pitch: u32,
        velocity: u32,
        state: &GRUState,
    ) -> Result<(Logits, GRUState)> {
        let device = state.h().device();
        let shape = (1, 1);
        let t = Tensor::from_slice(&[time], shape, device)?;
        let k = Tensor::from_slice(&[kind], shape, device)?;
        let p = Tensor::from_slice(&[pitch], shape, device)?;
        let v = Tensor::from_slice(&[velocity], shape, device)?;
        let embedded = self.embed(&t, &k, &p, &v)?;
        let flat = embedded.reshape((1, self.config.input_size()))?;
        let next = self.gru.step(&flat, state)?;
        let logits = self.heads(&next.h().reshape((1, 1, self.config.hidden))?)?;
        Ok((logits, next))
    }

    fn heads(&self, hidden: &Tensor) -> Result<Logits> {
        Ok(Logits {
            time: self.head_time.forward(hidden)?,
            kind: self.head_kind.forward(hidden)?,
            pitch: self.head_pitch.forward(hidden)?,
            velocity: self.head_velocity.forward(hidden)?,
        })
    }
}

/// Total number of trainable scalars in a variable map.
pub fn count_parameters(varmap: &candle_nn::VarMap) -> usize {
    varmap.all_vars().iter().map(|var| var.elem_count()).sum()
}

/// Build an empty model and its variable map on the CPU.
pub fn build(config: &ModelConfig) -> Result<(EventModel, candle_nn::VarMap)> {
    let varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let model = EventModel::new(vb, config)?;
    Ok((model, varmap))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_shapes_are_correct() {
        let (model, _varmap) = build(&ModelConfig {
            hidden: 16,
            embed: 8,
        })
        .expect("build");
        let b = 2;
        let t = 5;
        let time = Tensor::zeros((b, t), DType::U32, &Device::Cpu).unwrap();
        let kind = Tensor::zeros((b, t), DType::U32, &Device::Cpu).unwrap();
        let pitch = Tensor::zeros((b, t), DType::U32, &Device::Cpu).unwrap();
        let velocity = Tensor::zeros((b, t), DType::U32, &Device::Cpu).unwrap();
        let logits = model
            .forward(&time, &kind, &pitch, &velocity)
            .expect("forward");
        assert_eq!(logits.time.dims(), &[b, t, TIME_BINS]);
        assert_eq!(logits.kind.dims(), &[b, t, KIND_COUNT]);
        assert_eq!(logits.pitch.dims(), &[b, t, PITCH_COUNT]);
        assert_eq!(logits.velocity.dims(), &[b, t, VEL_BINS]);
    }

    #[test]
    fn step_matches_sequence_direction() {
        let (model, _varmap) = build(&ModelConfig {
            hidden: 16,
            embed: 8,
        })
        .expect("build");
        let state = model.zero_state(1).expect("state");
        let (logits, next) = model
            .step(0, crate::model::vocab::KIND_START, 0, 0, &state)
            .expect("step");
        assert_eq!(logits.pitch.dims(), &[1, 1, PITCH_COUNT]);
        assert_eq!(next.h().dims(), &[1, 16]);
    }
}
