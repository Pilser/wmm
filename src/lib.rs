//! World Music Model (WMM): symbolic music generation.
//!
//! The crate is split by responsibility:
//!
//! - [`events`] — the core `Note` / `NoteStream` types (seconds + MIDI numbers)
//! - [`midi`] — read and write Standard MIDI Files, tempo maps
//! - [`key`] — key detection and diatonic transposition
//! - [`harmonize`] — rule-based responses (the baseline before any model)
//! - [`render`] — MIDI → WAV with a SoundFont
//! - [`stats`] — descriptive statistics for datasets and debugging
//! - [`dataset`] — file discovery and split inference
//! - [`demo`] — a built-in phrase so the loop can be tested with no data
//! - [`cli`] — the `wmm` command-line interface

pub mod cli;
pub mod dataset;
pub mod demo;
pub mod events;
pub mod harmonize;
pub mod key;
pub mod midi;
pub mod model;
pub mod render;
pub mod stats;

pub use events::{Note, NoteStream};
