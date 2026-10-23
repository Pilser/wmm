//! Descriptive statistics over note streams. This is the "no notebook" tool:
//! it tells us whether a MIDI file parsed the way we expect.

use crate::events::{note_name, Note, NoteStream};

/// Statistics for one performance.
#[derive(Debug, Clone, Default)]
pub struct Analysis {
    pub notes: usize,
    pub duration_s: f64,
    pub min_pitch: Option<u8>,
    pub max_pitch: Option<u8>,
    pub mean_velocity: f64,
    /// Average number of notes sounding at the same time, weighted by time.
    pub mean_polyphony: f64,
}

impl Analysis {
    /// Notes per second of performance.
    pub fn density(&self) -> f64 {
        if self.duration_s <= 0.0 {
            0.0
        } else {
            self.notes as f64 / self.duration_s
        }
    }

    pub fn pitch_range(&self) -> String {
        match (self.min_pitch, self.max_pitch) {
            (Some(lo), Some(hi)) => format!("{}–{}", note_name(lo), note_name(hi)),
            _ => "–".to_string(),
        }
    }
}

/// Analyze a single note stream.
pub fn analyze(stream: &NoteStream) -> Analysis {
    if stream.is_empty() {
        return Analysis::default();
    }
    let duration = stream.duration();
    let pitches: Vec<u8> = stream.notes.iter().map(|n| n.pitch).collect();
    let velocity_sum: u64 = stream.notes.iter().map(|n| n.velocity as u64).sum();

    Analysis {
        notes: stream.len(),
        duration_s: duration,
        min_pitch: pitches.iter().copied().min(),
        max_pitch: pitches.iter().copied().max(),
        mean_velocity: velocity_sum as f64 / stream.len() as f64,
        mean_polyphony: mean_polyphony(&stream.notes, duration),
    }
}

/// Average polyphony: integrate the number of sounding notes over time, then
/// divide by total duration.
fn mean_polyphony(notes: &[Note], duration: f64) -> f64 {
    if duration <= 0.0 {
        return 0.0;
    }
    let mut points: Vec<(f64, i32)> = Vec::with_capacity(notes.len() * 2);
    for note in notes {
        points.push((note.start, 1));
        points.push((note.end().max(note.start), -1));
    }
    // At equal times, note-offs (-1) come first so a note's own start cannot
    // count as an overlap with itself.
    points.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    let mut area = 0.0;
    let mut current = 0i32;
    let mut last = 0.0;
    for (time, delta) in points {
        if time > last {
            area += current as f64 * (time - last);
            last = time;
        }
        current += delta;
    }
    area / duration
}

/// Aggregate statistics across many files.
#[derive(Debug, Clone, Default)]
pub struct Totals {
    pub files: usize,
    pub notes: usize,
    pub duration_s: f64,
    pub min_pitch: Option<u8>,
    pub max_pitch: Option<u8>,
    velocity_weighted_sum: f64,
    polyphony_weighted_sum: f64,
}

impl Totals {
    pub fn add(&mut self, analysis: &Analysis) {
        if analysis.notes == 0 {
            return;
        }
        self.files += 1;
        self.notes += analysis.notes;
        self.duration_s += analysis.duration_s;
        self.min_pitch = match (self.min_pitch, analysis.min_pitch) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        self.max_pitch = match (self.max_pitch, analysis.max_pitch) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        self.velocity_weighted_sum += analysis.mean_velocity * analysis.notes as f64;
        self.polyphony_weighted_sum += analysis.mean_polyphony * analysis.duration_s;
    }

    pub fn mean_velocity(&self) -> f64 {
        if self.notes == 0 {
            0.0
        } else {
            self.velocity_weighted_sum / self.notes as f64
        }
    }

    pub fn mean_polyphony(&self) -> f64 {
        if self.duration_s <= 0.0 {
            0.0
        } else {
            self.polyphony_weighted_sum / self.duration_s
        }
    }

    pub fn density(&self) -> f64 {
        if self.duration_s <= 0.0 {
            0.0
        } else {
            self.notes as f64 / self.duration_s
        }
    }

    pub fn pitch_range(&self) -> String {
        match (self.min_pitch, self.max_pitch) {
            (Some(lo), Some(hi)) => format!("{}–{}", note_name(lo), note_name(hi)),
            _ => "–".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{Note, NoteStream};

    #[test]
    fn empty_stream_is_zero() {
        let analysis = analyze(&NoteStream::new());
        assert_eq!(analysis.notes, 0);
        assert_eq!(analysis.duration_s, 0.0);
    }

    #[test]
    fn serial_notes_have_polyphony_one() {
        let stream = NoteStream::from_notes(vec![
            Note::new(0.0, 1.0, 60, 80, 0),
            Note::new(1.0, 1.0, 62, 90, 0),
        ]);
        let analysis = analyze(&stream);
        assert!((analysis.mean_polyphony - 1.0).abs() < 1e-9);
        assert!((analysis.duration_s - 2.0).abs() < 1e-9);
        assert!((analysis.density() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn chord_has_polyphony_three() {
        let stream = NoteStream::from_notes(vec![
            Note::new(0.0, 2.0, 60, 80, 0),
            Note::new(0.0, 2.0, 64, 80, 0),
            Note::new(0.0, 2.0, 67, 80, 0),
        ]);
        let analysis = analyze(&stream);
        assert!((analysis.mean_polyphony - 3.0).abs() < 1e-9);
    }

    #[test]
    fn totals_merge_files() {
        let a = analyze(&NoteStream::from_notes(vec![Note::new(
            0.0, 1.0, 60, 80, 0,
        )]));
        let b = analyze(&NoteStream::from_notes(vec![Note::new(
            0.0, 1.0, 72, 100, 0,
        )]));
        let mut totals = Totals::default();
        totals.add(&a);
        totals.add(&b);
        assert_eq!(totals.files, 2);
        assert_eq!(totals.notes, 2);
        assert!((totals.mean_velocity() - 90.0).abs() < 1e-9);
        assert_eq!(totals.pitch_range(), "C4–C5");
    }
}
