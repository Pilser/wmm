//! A built-in phrase, so the whole loop (events → MIDI → audio) can be tested
//! without downloading a dataset or owning a piano.
//!
//! The phrase is in A minor: an arpeggiated melody over a slow bass line.

use crate::events::{Note, NoteStream};
use crate::key::{Key, Mode};

/// Seconds per beat at the default 120 BPM used by the MIDI writer.
const BEAT: f64 = 0.5;

/// The key the phrase was written in. Hard-coded because we composed it; key
/// detection cannot tell A minor from its relative major, C major, because they
/// share the same seven notes.
pub fn key() -> Key {
    Key::new(9, Mode::Minor)
}

fn note(beat: f64, length_beats: f64, pitch: u8, velocity: u8, channel: u8) -> Note {
    Note::new(beat * BEAT, length_beats * BEAT, pitch, velocity, channel)
}

/// The demo phrase: two bars of melody plus a bass line.
pub fn phrase() -> NoteStream {
    let mut notes = Vec::new();

    // Melody (channel 0). A minor: A B C D E F G.
    let melody: &[(f64, f64, u8)] = &[
        (0.0, 0.5, 69), // A4
        (0.5, 0.5, 72), // C5
        (1.0, 1.0, 76), // E5
        (2.0, 0.5, 74), // D5
        (2.5, 0.5, 72), // C5
        (3.0, 1.0, 71), // B4
        (4.0, 0.5, 69), // A4
        (4.5, 0.5, 72), // C5
        (5.0, 1.0, 76), // E5
        (6.0, 0.5, 77), // F5
        (6.5, 0.5, 76), // E5
        (7.0, 1.0, 72), // C5
    ];
    for (start, length, pitch) in melody {
        notes.push(note(*start, *length, *pitch, 96, 0));
    }

    // Bass (channel 1): Am – F – C – G, one chord per two beats.
    let bass: &[(f64, u8)] = &[(0.0, 45), (2.0, 41), (4.0, 48), (6.0, 43)];
    for (start, pitch) in bass {
        notes.push(note(*start, 2.0, *pitch, 70, 1));
    }

    NoteStream::from_notes(notes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::analyze;

    #[test]
    fn demo_key_is_a_minor() {
        assert_eq!(key().name(), "A minor");
        assert_eq!(key().scale(), Mode::Minor.scale());
    }

    #[test]
    fn phrase_has_expected_shape() {
        let stream = phrase();
        assert_eq!(stream.len(), 16);
        let analysis = analyze(&stream);
        assert!((analysis.duration_s - 4.0).abs() < 1e-9);
        assert_eq!(analysis.min_pitch, Some(41));
        assert_eq!(analysis.max_pitch, Some(77));
        // Two voices at once for most of the phrase.
        assert!(analysis.mean_polyphony > 1.5);
    }
}
