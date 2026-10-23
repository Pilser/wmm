//! MIDI file support: reading, writing, and tempo handling.

pub mod read;
pub mod tempo;
pub mod write;

pub use read::{load, load_without_drums, parse_bytes, LoadedMidi};
pub use tempo::TempoMap;
pub use write::{build_smf, save, to_bytes, DEFAULT_PPQ};
