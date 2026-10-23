//! Core symbolic types: notes and note streams.
//!
//! Everything in this crate speaks in seconds and MIDI note numbers, not ticks.
//! Ticks are a file-format detail handled in [`crate::midi`].

use serde::{Deserialize, Serialize};

/// A single pitched note event.
///
/// `start` and `duration` are in seconds from the beginning of the performance.
/// `pitch` is a MIDI note number (0–127, 60 = middle C).
/// `velocity` is 1–127 (0 is reserved for "note off" on the wire).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Note {
    pub start: f64,
    pub duration: f64,
    pub pitch: u8,
    pub velocity: u8,
    pub channel: u8,
}

impl Note {
    pub fn new(start: f64, duration: f64, pitch: u8, velocity: u8, channel: u8) -> Self {
        Self {
            start,
            duration,
            pitch: pitch.min(127),
            velocity: velocity.clamp(1, 127),
            channel: channel.min(15),
        }
    }

    pub fn end(&self) -> f64 {
        self.start + self.duration
    }

    /// Pitch class 0–11, where 0 = C.
    pub fn pitch_class(&self) -> u8 {
        self.pitch % 12
    }

    /// Scientific octave number, where middle C (60) is C4.
    pub fn octave(&self) -> i32 {
        self.pitch as i32 / 12 - 1
    }

    /// Human-readable name such as `C4` or `F#5`.
    pub fn name(&self) -> String {
        note_name(self.pitch)
    }
}

/// A time-ordered collection of notes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NoteStream {
    pub notes: Vec<Note>,
}

impl NoteStream {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_notes(mut notes: Vec<Note>) -> Self {
        notes.sort_by(|a, b| {
            a.start
                .partial_cmp(&b.start)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.pitch.cmp(&b.pitch))
        });
        Self { notes }
    }

    pub fn push(&mut self, note: Note) {
        self.notes.push(note);
    }

    /// Sort by start time. Call after building a stream by hand.
    pub fn sort(&mut self) {
        self.notes.sort_by(|a, b| {
            a.start
                .partial_cmp(&b.start)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.pitch.cmp(&b.pitch))
        });
    }

    pub fn len(&self) -> usize {
        self.notes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }

    /// Time of the last note-off, or 0.0 for an empty stream.
    pub fn duration(&self) -> f64 {
        self.notes.iter().map(Note::end).fold(0.0, f64::max)
    }

    /// Merge another stream into this one, keeping the result sorted.
    pub fn extend(&mut self, other: &NoteStream) {
        self.notes.extend_from_slice(&other.notes);
        self.sort();
    }

    /// Shift every note in time by `seconds` (negative values move earlier).
    pub fn shifted(&self, seconds: f64) -> NoteStream {
        let mut out = self.clone();
        for note in &mut out.notes {
            note.start = (note.start + seconds).max(0.0);
        }
        out
    }

    /// Keep only notes whose start lies in `[from, to)`.
    pub fn slice(&self, from: f64, to: f64) -> NoteStream {
        NoteStream::from_notes(
            self.notes
                .iter()
                .copied()
                .filter(|n| n.start >= from && n.start < to)
                .collect(),
        )
    }
}

/// Convert a MIDI note number to its name, e.g. `60 -> "C4"`, `61 -> "C#4"`.
pub fn note_name(pitch: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = pitch as i32 / 12 - 1;
    format!("{}{}", NAMES[(pitch % 12) as usize], octave)
}

/// Frequency in hertz of a MIDI note number in equal temperament (A4 = 440 Hz).
pub fn pitch_to_hz(pitch: u8) -> f64 {
    440.0 * 2f64.powf((pitch as f64 - 69.0) / 12.0)
}

/// Nearest MIDI note number for a frequency in hertz.
pub fn hz_to_pitch(hz: f64) -> u8 {
    if !(hz.is_finite() && hz > 0.0) {
        return 0;
    }
    let value = 69.0 + 12.0 * (hz / 440.0).log2();
    value.round().clamp(0.0, 127.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn middle_c_is_60() {
        assert_eq!(hz_to_pitch(261.63), 60);
        assert_eq!(pitch_to_hz(69), 440.0);
        assert_eq!(note_name(60), "C4");
        assert_eq!(note_name(61), "C#4");
    }

    #[test]
    fn stream_sorts_and_measures() {
        let stream = NoteStream::from_notes(vec![
            Note::new(1.0, 0.5, 64, 90, 0),
            Note::new(0.0, 2.0, 60, 80, 0),
        ]);
        assert_eq!(stream.notes[0].pitch, 60);
        assert!((stream.duration() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn octave_and_pitch_class() {
        let note = Note::new(0.0, 1.0, 60, 100, 0);
        assert_eq!(note.octave(), 4);
        assert_eq!(note.pitch_class(), 0);
    }
}
