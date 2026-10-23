//! Neural model: vocabulary, corpus, GRU, training, sampling.

pub mod corpus;
pub mod gru;
pub mod merge;
pub mod rng;
pub mod sample;
pub mod train;
pub mod vocab;

pub use gru::{build, count_parameters, EventModel, ModelConfig};
pub use merge::{merge, MergeReport};
pub use sample::{generate, SampleConfig};
pub use train::{load_checkpoint, run, CheckpointInfo, TrainConfig, TrainReport};
