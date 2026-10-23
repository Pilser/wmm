//! Musical keys: detection and diatonic transposition.
//!
//! "Diatonic" means *within the scale*: a third above C in C major is E, but a
//! third above D is F. That is different from a fixed semitone shift, and it is
//! what makes rule-based harmony sound musical.

use anyhow::{bail, Result};

use crate::events::Note;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Major,
    Minor,
}

impl Mode {
    /// Semitone offsets of the seven scale degrees from the tonic.
    pub fn scale(self) -> [i32; 7] {
        match self {
            Mode::Major => [0, 2, 4, 5, 7, 9, 11],
            Mode::Minor => [0, 2, 3, 5, 7, 8, 10],
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Major => "major",
            Mode::Minor => "minor",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Key {
    /// Pitch class of the tonic: 0 = C, 1 = C#/Db, ... 11 = B.
    pub tonic: u8,
    pub mode: Mode,
}

impl Key {
    pub fn new(tonic: u8, mode: Mode) -> Self {
        Self {
            tonic: tonic % 12,
            mode,
        }
    }

    /// C major, the default when nothing else is known.
    pub fn c_major() -> Self {
        Self::new(0, Mode::Major)
    }

    pub fn scale(self) -> [i32; 7] {
        self.mode.scale()
    }

    /// Human-readable name, e.g. `F# minor`.
    pub fn name(self) -> String {
        const NAMES: [&str; 12] = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        format!("{} {}", NAMES[self.tonic as usize], self.mode.as_str())
    }

    /// Parse forms like `C`, `c:maj`, `F# minor`, `Am`, `a:min`.
    pub fn parse(text: &str) -> Result<Self> {
        let cleaned: String = text
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .to_lowercase();
        if cleaned.is_empty() {
            bail!("empty key; expected something like C, F# minor, or Am");
        }

        let mut chars = cleaned.chars();
        let letter = chars.next().unwrap();
        let mut tonic = match letter {
            'c' => 0,
            'd' => 2,
            'e' => 4,
            'f' => 5,
            'g' => 7,
            'a' => 9,
            'b' => 11,
            _ => bail!("invalid key root '{letter}'; use A–G"),
        };

        let mut rest: String = chars.collect();
        if let Some(first) = rest.chars().next() {
            match first {
                '#' | 's' => {
                    tonic = (tonic + 1) % 12;
                    rest.remove(0);
                }
                'b' if rest.len() > 1 => {
                    // "bb" = B flat; a lone trailing "b" would be ambiguous, so
                    // require more characters (e.g. "bbmaj").
                    tonic = (tonic + 11) % 12;
                    rest.remove(0);
                }
                _ => {}
            }
        }

        let mode = match rest.trim_start_matches(':') {
            "" | "maj" | "major" => Mode::Major,
            "m" | "min" | "minor" => Mode::Minor,
            other => bail!("invalid key quality '{other}'; use maj, min, or m"),
        };
        Ok(Self::new(tonic, mode))
    }

    /// Estimate the key from a note stream using duration-weighted pitch-class
    /// histograms and the Krumhansl–Kessler profiles.
    pub fn detect(notes: &[Note]) -> Self {
        let mut histogram = [0.0f64; 12];
        for note in notes {
            histogram[(note.pitch % 12) as usize] += note.duration.max(0.01);
        }
        if histogram.iter().all(|v| *v == 0.0) {
            return Self::c_major();
        }

        let mut best = (f64::NEG_INFINITY, Self::c_major());
        for tonic in 0..12u8 {
            for mode in [Mode::Major, Mode::Minor] {
                let key = Self::new(tonic, mode);
                let correlation = correlation(&histogram, &profile_for(key));
                if correlation > best.0 {
                    best = (correlation, key);
                }
            }
        }
        best.1
    }
}

/// Krumhansl–Kessler key profiles (listener ratings of how well each pitch class
/// fits a key), ordered from the tonic upwards.
const MAJOR_PROFILE: [f64; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const MINOR_PROFILE: [f64; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

fn profile_for(key: Key) -> [f64; 12] {
    let base = match key.mode {
        Mode::Major => MAJOR_PROFILE,
        Mode::Minor => MINOR_PROFILE,
    };
    let mut out = [0.0; 12];
    for (pc, value) in base.iter().enumerate() {
        out[(pc + key.tonic as usize) % 12] = *value;
    }
    out
}

fn correlation(a: &[f64; 12], b: &[f64; 12]) -> f64 {
    let mean_a = a.iter().sum::<f64>() / 12.0;
    let mean_b = b.iter().sum::<f64>() / 12.0;
    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;
    for i in 0..12 {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    if den_a <= 0.0 || den_b <= 0.0 {
        0.0
    } else {
        num / (den_a.sqrt() * den_b.sqrt())
    }
}

/// Move `pitch` by `degrees` scale steps within `key`.
///
/// `degrees = 1` is the next scale tone up, `2` is a diatonic third,
/// `4` a diatonic fifth, `7` an octave. Negative values move down.
/// Chromatic notes snap to the scale tone below before shifting.
pub fn diatonic_shift(key: Key, pitch: u8, degrees: i32) -> u8 {
    if degrees == 0 {
        return pitch;
    }
    let scale = key.scale();
    let absolute = pitch as i32 - key.tonic as i32;
    let octave = absolute.div_euclid(12);
    let pitch_class = absolute.rem_euclid(12);

    // Find the scale degree at or below this pitch class.
    let degree = scale
        .iter()
        .rposition(|offset| *offset <= pitch_class)
        .unwrap_or(0);

    let target = degree as i32 + degrees;
    let target_octave = octave + target.div_euclid(7);
    let target_degree = target.rem_euclid(7) as usize;
    let shifted = key.tonic as i32 + target_octave * 12 + scale[target_degree];
    shifted.clamp(0, 127) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::Note;

    #[test]
    fn parse_common_key_names() {
        assert_eq!(Key::parse("C").unwrap(), Key::new(0, Mode::Major));
        assert_eq!(Key::parse("a:maj").unwrap(), Key::new(9, Mode::Major));
        assert_eq!(Key::parse("F# minor").unwrap(), Key::new(6, Mode::Minor));
        assert_eq!(Key::parse("Am").unwrap(), Key::new(9, Mode::Minor));
        assert_eq!(Key::parse("bbmaj").unwrap(), Key::new(10, Mode::Major));
        assert!(Key::parse("H").is_err());
        assert!(Key::parse("C weird").is_err());
    }

    #[test]
    fn thirds_stay_in_key() {
        let c = Key::c_major();
        // C4 -> E4, D4 -> F4, E4 -> G4
        assert_eq!(diatonic_shift(c, 60, 2), 64);
        assert_eq!(diatonic_shift(c, 62, 2), 65);
        assert_eq!(diatonic_shift(c, 64, 2), 67);
    }

    #[test]
    fn fifth_and_octave() {
        let c = Key::c_major();
        assert_eq!(diatonic_shift(c, 60, 4), 67);
        assert_eq!(diatonic_shift(c, 60, 7), 72);
        assert_eq!(diatonic_shift(c, 60, -7), 48);
    }

    #[test]
    fn detects_c_major_triad_heavy_music() {
        let mut notes = Vec::new();
        for i in 0..8 {
            for pitch in [60, 64, 67] {
                notes.push(Note::new(i as f64, 0.5, pitch, 90, 0));
            }
        }
        assert_eq!(Key::detect(&notes).tonic, 0);
        assert_eq!(Key::detect(&notes).mode, Mode::Major);
    }

    #[test]
    fn detects_a_minor() {
        let mut notes = Vec::new();
        for i in 0..8 {
            for pitch in [57, 60, 64] {
                notes.push(Note::new(i as f64, 0.5, pitch, 90, 0));
            }
        }
        let key = Key::detect(&notes);
        assert_eq!(key.tonic, 9);
        assert_eq!(key.mode, Mode::Minor);
    }
}
