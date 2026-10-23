//! Rule-based response generation: the "physics" before the "taste".
//!
//! These rules take notes that were played and return notes to play back. They
//! are deliberately simple and deterministic, so they form a baseline that a
//! learned model must beat — and a correctness test for the MIDI pipeline.

use std::fmt;

use anyhow::{bail, Result};

use crate::events::{Note, NoteStream};
use crate::key::{diatonic_shift, Key};

/// What kind of response to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Same pitch, same octave: proves the listen → play loop.
    Unison,
    /// One octave up, within the key.
    Octave,
    /// A diatonic third above.
    Third,
    /// A diatonic fourth above.
    Fourth,
    /// A diatonic fifth above.
    Fifth,
    /// A diatonic sixth above.
    Sixth,
    /// Repeat the input after a delay, softer.
    Echo,
}

impl Mode {
    /// Scale-degree offset used by this mode, if any.
    pub fn degrees(self) -> Option<i32> {
        match self {
            Mode::Unison => Some(0),
            Mode::Octave => Some(7),
            Mode::Third => Some(2),
            Mode::Fourth => Some(3),
            Mode::Fifth => Some(4),
            Mode::Sixth => Some(5),
            Mode::Echo => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Unison => "unison",
            Mode::Octave => "octave",
            Mode::Third => "third",
            Mode::Fourth => "fourth",
            Mode::Fifth => "fifth",
            Mode::Sixth => "sixth",
            Mode::Echo => "echo",
        }
    }

    pub fn parse(text: &str) -> Result<Self> {
        match text.to_lowercase().as_str() {
            "unison" | "same" => Ok(Mode::Unison),
            "octave" | "oct" => Ok(Mode::Octave),
            "third" | "3rd" => Ok(Mode::Third),
            "fourth" | "4th" => Ok(Mode::Fourth),
            "fifth" | "5th" => Ok(Mode::Fifth),
            "sixth" | "6th" => Ok(Mode::Sixth),
            "echo" => Ok(Mode::Echo),
            other => bail!(
                "unknown mode '{other}'; expected unison, octave, third, fourth, fifth, sixth, or echo"
            ),
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parameters for [`harmonize`].
#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub mode: Mode,
    /// Key used for diatonic shifts. If `None`, harmony is computed in C major.
    pub key: Option<Key>,
    /// Velocity multiplier for the generated notes, 0.0–1.0.
    pub velocity_scale: f32,
    /// Delay in seconds for [`Mode::Echo`].
    pub echo_delay: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            mode: Mode::Third,
            key: None,
            velocity_scale: 0.85,
            echo_delay: 0.35,
        }
    }
}

impl Options {
    pub fn with_mode(mode: Mode) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }
}

/// Generate a response stream. The input is **not** included in the result.
pub fn harmonize(input: &NoteStream, options: &Options) -> NoteStream {
    let key = options.key.unwrap_or_else(Key::c_major);
    let scale = options.velocity_scale.clamp(0.0, 1.0);

    let response = match options.mode {
        Mode::Echo => input.shifted(options.echo_delay),
        _ => {
            let degrees = options.mode.degrees().unwrap_or(0);
            NoteStream::from_notes(
                input
                    .notes
                    .iter()
                    .map(|note| {
                        let pitch = diatonic_shift(key, note.pitch, degrees);
                        Note::new(
                            note.start,
                            note.duration,
                            pitch,
                            scale_velocity(note.velocity, scale),
                            note.channel,
                        )
                    })
                    .collect(),
            )
        }
    };

    if options.mode == Mode::Echo {
        NoteStream::from_notes(
            response
                .notes
                .into_iter()
                .map(|mut note| {
                    note.velocity = scale_velocity(note.velocity, scale);
                    note
                })
                .collect(),
        )
    } else {
        response
    }
}

/// Put the response on a separate MIDI channel so a DAW can give it its own
/// instrument.
pub fn assign_channel(stream: &NoteStream, channel: u8) -> NoteStream {
    NoteStream::from_notes(
        stream
            .notes
            .iter()
            .map(|note| Note {
                channel: channel.min(15),
                ..*note
            })
            .collect(),
    )
}

fn scale_velocity(velocity: u8, scale: f32) -> u8 {
    ((velocity as f32 * scale).round() as i32).clamp(1, 127) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{Note, NoteStream};

    fn input() -> NoteStream {
        NoteStream::from_notes(vec![
            Note::new(0.0, 0.5, 60, 100, 0),
            Note::new(0.5, 0.5, 62, 100, 0),
        ])
    }

    #[test]
    fn unison_keeps_pitches() {
        let out = harmonize(&input(), &Options::with_mode(Mode::Unison));
        assert_eq!(out.notes[0].pitch, 60);
        assert_eq!(out.notes[1].pitch, 62);
    }

    #[test]
    fn third_is_diatonic() {
        let out = harmonize(&input(), &Options::with_mode(Mode::Third));
        assert_eq!(out.notes[0].pitch, 64); // C->E
        assert_eq!(out.notes[1].pitch, 65); // D->F
    }

    #[test]
    fn octave_is_twelve_semitones() {
        let out = harmonize(&input(), &Options::with_mode(Mode::Octave));
        assert_eq!(out.notes[0].pitch, 72);
    }

    #[test]
    fn echo_delays_and_softens() {
        let options = Options {
            mode: Mode::Echo,
            echo_delay: 1.0,
            ..Options::default()
        };
        let out = harmonize(&input(), &options);
        assert!((out.notes[0].start - 1.0).abs() < 1e-9);
        assert!(out.notes[0].velocity < 100);
    }

    #[test]
    fn mode_names_roundtrip() {
        for name in [
            "unison", "octave", "third", "fourth", "fifth", "sixth", "echo",
        ] {
            assert_eq!(Mode::parse(name).unwrap().as_str(), name);
        }
        assert!(Mode::parse("seventh").is_err());
    }
}
