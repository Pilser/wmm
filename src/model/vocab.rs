//! Event vocabulary: notes become a discrete sequence of symbols.
//!
//! The model never sees floating-point timing. Each step is one event:
//!
//! ```text
//! (delta-time bucket, kind, pitch, velocity bucket)
//! ```
//!
//! `kind` is one of start / note-on / note-off. Delta time is quantized on a
//! logarithmic grid: fine near zero (where rhythm lives) and coarse far away.
//! This is the same idea as Performance RNN's time-shift tokens.

use crate::events::{Note, NoteStream};

/// Number of log-spaced delta-time bins.
pub const TIME_BINS: usize = 128;
/// Kinds of event: start of sequence, note-on, note-off.
pub const KIND_COUNT: usize = 3;
/// MIDI note numbers 0–127.
pub const PITCH_COUNT: usize = 128;
/// Velocity is quantized to 32 buckets to keep the output head small.
pub const VEL_BINS: usize = 32;

/// Smallest non-zero time step, in seconds (4 ms). Fast enough for trills,
/// slow enough that a 128-bin log grid covers ~2.3 hours.
pub const TIME_FLOOR: f64 = 0.004;
/// Bins per doubling of the time step. Six gives ~12 % time resolution.
pub const TIME_BINS_PER_OCTAVE: f64 = 6.0;

pub const KIND_START: u32 = 0;
pub const KIND_ON: u32 = 1;
pub const KIND_OFF: u32 = 2;

/// One discrete event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Event {
    /// Seconds since the previous event.
    pub dt: f64,
    /// 0 = start, 1 = note-on, 2 = note-off.
    pub kind: u32,
    pub pitch: u8,
    /// 1–127 for note-on; 0 for note-off and start.
    pub velocity: u8,
}

impl Event {
    pub fn start() -> Self {
        Self {
            dt: 0.0,
            kind: KIND_START,
            pitch: 0,
            velocity: 0,
        }
    }

    pub fn dt_bin(&self) -> u32 {
        quantize_dt(self.dt)
    }

    pub fn vel_bin(&self) -> u32 {
        quantize_velocity(self.velocity)
    }

    /// Rebuild an event from model output.
    pub fn from_bins(dt_bin: u32, kind: u32, pitch: u32, vel_bin: u32) -> Self {
        let velocity = if kind == KIND_ON {
            dequantize_velocity(vel_bin)
        } else {
            0
        };
        Self {
            dt: dequantize_dt(dt_bin),
            kind,
            pitch: (pitch as u8).min(127),
            velocity,
        }
    }
}

/// Quantize a delta time in seconds to a bin.
pub fn quantize_dt(dt: f64) -> u32 {
    if !dt.is_finite() || dt <= TIME_FLOOR {
        return 0;
    }
    let steps = (dt / TIME_FLOOR).log2() * TIME_BINS_PER_OCTAVE;
    (1.0 + steps.round()).clamp(1.0, (TIME_BINS - 1) as f64) as u32
}

/// Inverse of [`quantize_dt`].
pub fn dequantize_dt(bin: u32) -> f64 {
    if bin == 0 {
        return 0.0;
    }
    let steps = (bin.min(TIME_BINS as u32 - 1) as f64 - 1.0) / TIME_BINS_PER_OCTAVE;
    TIME_FLOOR * 2f64.powf(steps)
}

/// Quantize MIDI velocity (1–127) to 32 buckets.
pub fn quantize_velocity(velocity: u8) -> u32 {
    if velocity == 0 {
        return 0;
    }
    (((velocity as u32 - 1) * VEL_BINS as u32) / 128).min(VEL_BINS as u32 - 1)
}

/// Inverse of [`quantize_velocity`], always at least 1.
pub fn dequantize_velocity(bin: u32) -> u8 {
    let bin = bin.min(VEL_BINS as u32 - 1);
    (((bin * 4) + 2).clamp(1, 127)) as u8
}

/// Convert a note stream into a time-ordered event sequence with a leading
/// start token. At equal times, note-offs come before note-ons so a repeated
/// pitch is released before it is struck again.
pub fn stream_to_events(stream: &NoteStream) -> Vec<Event> {
    let mut raw: Vec<(f64, u32, u8, u8)> = Vec::with_capacity(stream.len() * 2 + 1);
    for note in &stream.notes {
        raw.push((note.start, KIND_ON, note.pitch, note.velocity.clamp(1, 127)));
        raw.push((note.end(), KIND_OFF, note.pitch, 0));
    }
    raw.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.cmp(&a.1))
            .then(a.2.cmp(&b.2))
    });

    let mut events = Vec::with_capacity(raw.len() + 1);
    events.push(Event::start());
    let mut previous = 0.0;
    for (time, kind, pitch, velocity) in raw {
        let dt = (time - previous).max(0.0);
        events.push(Event {
            dt,
            kind,
            pitch,
            velocity,
        });
        previous = time;
    }
    events
}

/// Convert events back into notes. Unmatched note-offs are ignored; notes still
/// open at the end are closed after `tail` seconds.
pub fn events_to_stream(events: &[Event], tail: f64) -> NoteStream {
    use std::collections::HashMap;

    let mut notes = Vec::new();
    let mut open: HashMap<u8, (f64, u8)> = HashMap::new();
    let mut time = 0.0;
    for event in events {
        time += event.dt;
        match event.kind {
            KIND_ON if event.velocity > 0 => {
                open.insert(event.pitch, (time, event.velocity));
            }
            KIND_OFF => {
                if let Some((start, velocity)) = open.remove(&event.pitch) {
                    notes.push(Note::new(
                        start,
                        (time - start).max(TIME_FLOOR),
                        event.pitch,
                        velocity,
                        0,
                    ));
                }
            }
            _ => {}
        }
    }
    for (pitch, (start, velocity)) in open {
        notes.push(Note::new(start, tail, pitch, velocity, 0));
    }
    NoteStream::from_notes(notes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_bins_round_trip_within_a_twelfth() {
        for dt in [0.001, 0.004, 0.02, 0.1, 0.5, 1.0, 3.0, 30.0] {
            let back = dequantize_dt(quantize_dt(dt));
            if dt <= TIME_FLOOR {
                assert_eq!(back, 0.0);
            } else {
                let ratio = back / dt;
                assert!(ratio > 0.9 && ratio < 1.1, "dt {dt} -> {back}");
            }
        }
    }

    #[test]
    fn time_bins_are_monotonic() {
        let mut previous = 0;
        for step in 0..200 {
            let dt = 0.004 * 1.1f64.powi(step);
            let bin = quantize_dt(dt);
            assert!(bin >= previous);
            previous = bin;
        }
    }

    #[test]
    fn velocity_round_trip_is_close() {
        for velocity in [1u8, 20, 64, 100, 127] {
            let back = dequantize_velocity(quantize_velocity(velocity));
            assert!(
                (back as i32 - velocity as i32).abs() <= 2,
                "{velocity} -> {back}"
            );
        }
    }

    #[test]
    fn events_keep_note_count_and_order() {
        let stream = NoteStream::from_notes(vec![
            Note::new(0.0, 0.5, 60, 100, 0),
            Note::new(0.5, 0.25, 62, 80, 0),
        ]);
        let events = stream_to_events(&stream);
        assert_eq!(events[0].kind, KIND_START);
        assert_eq!(events.len(), 5);
        let back = events_to_stream(&events, 0.5);
        assert_eq!(back.len(), 2);
        assert_eq!(back.notes[0].pitch, 60);
    }
}
