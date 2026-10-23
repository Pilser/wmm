//! Corpus building: MIDI files become fixed-length event windows.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use crate::dataset;
use crate::midi;
use crate::model::vocab::{stream_to_events, Event};

/// Which part of a manifest (CSV index) to train on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorpusSplit {
    Train,
    Validation,
    Test,
    All,
}

impl CorpusSplit {
    fn requires_manifest(self) -> bool {
        !matches!(self, CorpusSplit::All)
    }
}

/// Parsed performances, ready to be cut into windows.
#[derive(Debug, Clone, Default)]
pub struct Corpus {
    /// One event sequence per file, each starting with a start token.
    pub sequences: Vec<Vec<Event>>,
    pub files: usize,
    pub events: usize,
}

impl Corpus {
    /// Load up to `limit` MIDI files from a file or directory.
    pub fn load(path: &Path, limit: Option<usize>) -> Result<Self> {
        Self::load_split(path, limit, CorpusSplit::All)
    }

    /// Load up to `limit` MIDI files, restricted to one manifest split when
    /// `split` is not [`CorpusSplit::All`]. The manifest is the MAESTRO-style
    /// `*.csv` index next to the files.
    pub fn load_split(path: &Path, limit: Option<usize>, split: CorpusSplit) -> Result<Self> {
        let mut paths = dataset::find_midi_files(path)?;
        if paths.is_empty() {
            bail!("no MIDI files found under {}", path.display());
        }
        if split.requires_manifest() {
            let manifest_path = dataset::find_manifest(path).ok_or_else(|| {
                anyhow::anyhow!(
                    "split filtering needs a MAESTRO-style CSV index next to the files; none found under {}",
                    path.display()
                )
            })?;
            let manifest = dataset::Manifest::load(&manifest_path)?;
            let wanted = match split {
                CorpusSplit::Train => dataset::Split::Train,
                CorpusSplit::Validation => dataset::Split::Validation,
                CorpusSplit::Test => dataset::Split::Test,
                CorpusSplit::All => unreachable!(),
            };
            let before = paths.len();
            paths.retain(|p| manifest.split_for(p) == wanted);
            if paths.is_empty() {
                bail!(
                    "no files in split {} under {} ({} files looked at)",
                    wanted.as_str(),
                    path.display(),
                    before
                );
            }
        }
        if let Some(limit) = limit {
            paths.truncate(limit);
        }

        let mut corpus = Corpus::default();
        for path in paths {
            let loaded = midi::load(&path)?;
            if loaded.stream.is_empty() {
                continue;
            }
            let events = stream_to_events(&loaded.stream);
            corpus.events += events.len();
            corpus.sequences.push(events);
            corpus.files += 1;
        }
        if corpus.files == 0 {
            bail!("all files were empty under {}", path.display());
        }
        Ok(corpus)
    }

    /// Cut every sequence into windows of exactly `length + 1` events.
    ///
    /// The first `length` events are model input; the last is the target. Every
    /// window starts with a start token so the recurrent state has a clean
    /// beginning.
    pub fn windows(&self, length: usize, stride: usize) -> Vec<Vec<Event>> {
        if length == 0 {
            return Vec::new();
        }
        let stride = stride.max(1);
        let mut out = Vec::new();
        for sequence in &self.sequences {
            if sequence.len() < 2 {
                continue;
            }
            let mut start = 0;
            while start + length < sequence.len() {
                let mut window = Vec::with_capacity(length + 1);
                window.push(Event::start());
                window.extend_from_slice(&sequence[start..start + length]);
                out.push(window);
                start += stride;
            }
        }
        out
    }
}

/// List files without parsing them (used for progress messages).
pub fn peek_files(path: &Path, limit: Option<usize>) -> Result<Vec<PathBuf>> {
    let mut paths = dataset::find_midi_files(path)?;
    if let Some(limit) = limit {
        paths.truncate(limit);
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_the_requested_length() {
        let mut corpus = Corpus::default();
        corpus.sequences.push(vec![Event::start(); 10]);
        let windows = corpus.windows(4, 2);
        assert!(!windows.is_empty());
        for window in &windows {
            assert_eq!(window.len(), 5);
            assert_eq!(window[0].kind, crate::model::vocab::KIND_START);
        }
    }

    #[test]
    fn short_sequences_produce_no_windows() {
        let mut corpus = Corpus::default();
        corpus.sequences.push(vec![Event::start(); 3]);
        assert!(corpus.windows(8, 4).is_empty());
    }
}
