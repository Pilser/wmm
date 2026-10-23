//! Read a Standard MIDI File into a [`NoteStream`].
//!
//! Ticks are converted to seconds using the file's tempo map, so every note
//! carries absolute time. Notes are matched on (channel, pitch) with a stack, so
//! repeated notes of the same pitch do not steal each other's note-off.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};

use crate::events::{Note, NoteStream};
use crate::midi::tempo::TempoMap;

/// A MIDI file after parsing.
#[derive(Debug, Clone)]
pub struct LoadedMidi {
    pub stream: NoteStream,
    pub tempo: TempoMap,
    /// Track name from the first track that declares one.
    pub name: Option<String>,
    pub track_count: usize,
}

/// Read and parse a `.mid` / `.midi` file.
pub fn load(path: &Path) -> Result<LoadedMidi> {
    let bytes =
        std::fs::read(path).with_context(|| format!("cannot read MIDI file {}", path.display()))?;
    parse_bytes(&bytes).with_context(|| format!("cannot parse MIDI file {}", path.display()))
}

/// Parse raw Standard MIDI File bytes.
pub fn parse_bytes(bytes: &[u8]) -> Result<LoadedMidi> {
    let smf = Smf::parse(bytes).context("invalid Standard MIDI File")?;
    let ppq = match smf.header.timing {
        Timing::Metrical(ppq) => ppq.as_int(),
        Timing::Timecode(_, _) => {
            bail!("SMPTE timecode MIDI files are not supported; export with metrical timing")
        }
    };

    // Pass 1: collect tempo changes from all tracks.
    let mut tempo = TempoMap::new(ppq);
    for track in &smf.tracks {
        let mut tick = 0u64;
        for event in track {
            tick += event.delta.as_int() as u64;
            if let TrackEventKind::Meta(MetaMessage::Tempo(us)) = event.kind {
                tempo.push(tick, us.as_int());
            }
        }
    }
    tempo.finalize();

    // Pass 2: collect notes and the first track name.
    let mut name = None;
    let mut notes = Vec::new();
    for track in &smf.tracks {
        let mut tick = 0u64;
        let mut open: HashMap<(u8, u8), Vec<(u64, u8)>> = HashMap::new();
        for event in track {
            tick += event.delta.as_int() as u64;
            match event.kind {
                TrackEventKind::Meta(MetaMessage::TrackName(raw)) => {
                    if name.is_none() {
                        name = Some(String::from_utf8_lossy(raw).trim().to_string());
                    }
                }
                TrackEventKind::Midi { channel, message } => match message {
                    MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                        open.entry((channel.as_int(), key.as_int()))
                            .or_default()
                            .push((tick, vel.as_int()));
                    }
                    MidiMessage::NoteOn { key, vel: _ } | MidiMessage::NoteOff { key, .. } => {
                        let slot = (channel.as_int(), key.as_int());
                        if let Some(stack) = open.get_mut(&slot) {
                            if let Some((start_tick, velocity)) = stack.pop() {
                                notes.push(Note::new(
                                    tempo.seconds_at(start_tick),
                                    tempo.seconds_at(tick) - tempo.seconds_at(start_tick),
                                    slot.1,
                                    velocity,
                                    slot.0,
                                ));
                            }
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        // Notes still held at end of track: close them at the last tick seen.
        for ((channel, pitch), stack) in open {
            for (start_tick, velocity) in stack {
                notes.push(Note::new(
                    tempo.seconds_at(start_tick),
                    (tempo.seconds_at(tick) - tempo.seconds_at(start_tick)).max(0.01),
                    pitch,
                    velocity,
                    channel,
                ));
            }
        }
    }

    Ok(LoadedMidi {
        stream: NoteStream::from_notes(notes),
        tempo,
        name: name.filter(|n| !n.is_empty()),
        track_count: smf.tracks.len(),
    })
}

/// Parse a file and keep only channels 0–8 and 10–15 (drops the drum channel 9).
pub fn load_without_drums(path: &Path) -> Result<LoadedMidi> {
    let mut loaded = load(path)?;
    loaded.stream.notes.retain(|n| n.channel != 9);
    Ok(loaded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi::write::{build_smf, to_bytes};

    fn sample_bytes() -> Vec<u8> {
        let stream = NoteStream::from_notes(vec![
            Note::new(0.0, 0.5, 60, 100, 0),
            Note::new(0.5, 1.0, 67, 90, 0),
            Note::new(0.5, 1.0, 64, 80, 0),
        ]);
        let smf = build_smf(&stream, 480).expect("build smf");
        to_bytes(&smf).expect("serialize smf")
    }

    #[test]
    fn roundtrip_notes_and_times() {
        let loaded = parse_bytes(&sample_bytes()).expect("parse");
        assert_eq!(loaded.stream.len(), 3);
        assert_eq!(loaded.stream.notes[0].pitch, 60);
        assert!((loaded.stream.notes[0].start - 0.0).abs() < 1e-6);
        // Second and third notes start on beat 2 (0.5 s at 120 BPM).
        assert!((loaded.stream.notes[1].start - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sustained_notes_are_closed_at_end() {
        let bytes = sample_bytes();
        let loaded = parse_bytes(&bytes).expect("parse");
        assert!(loaded.stream.duration() >= 1.4);
    }
}
