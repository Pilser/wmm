//! Tempo maps: convert MIDI ticks to seconds.
//!
//! A Standard MIDI File stores time in *ticks*, and the tempo tells us how many
//! ticks fit into a second. Tempo can change mid-file, so we keep a list of
//! change points and integrate across them.

/// Default tempo: 120 beats per minute = 500,000 microseconds per quarter note.
pub const DEFAULT_US_PER_QUARTER: u32 = 500_000;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Change {
    tick: u64,
    us_per_quarter: u32,
}

/// Piecewise-constant tempo map for one MIDI file.
#[derive(Debug, Clone, PartialEq)]
pub struct TempoMap {
    ppq: u16,
    changes: Vec<Change>,
}

impl TempoMap {
    /// Create a map at the default 120 BPM tempo.
    pub fn new(ppq: u16) -> Self {
        Self {
            ppq: ppq.max(1),
            changes: vec![Change {
                tick: 0,
                us_per_quarter: DEFAULT_US_PER_QUARTER,
            }],
        }
    }

    pub fn ppq(&self) -> u16 {
        self.ppq
    }

    /// Add a tempo change. Order does not matter; call [`TempoMap::finalize`] once
    /// all changes are known.
    pub fn push(&mut self, tick: u64, us_per_quarter: u32) {
        self.changes.push(Change {
            tick,
            us_per_quarter: us_per_quarter.max(1),
        });
    }

    /// Sort and de-duplicate tempo changes. Also guarantees a change at tick 0.
    pub fn finalize(&mut self) {
        self.changes.sort_by_key(|c| c.tick);
        self.changes.dedup_by_key(|c| c.tick);
        if self.changes.first().map(|c| c.tick) != Some(0) {
            self.changes.insert(
                0,
                Change {
                    tick: 0,
                    us_per_quarter: DEFAULT_US_PER_QUARTER,
                },
            );
        }
    }

    /// Seconds elapsed from tick 0 to `tick`.
    pub fn seconds_at(&self, tick: u64) -> f64 {
        let mut seconds = 0.0;
        for (i, change) in self.changes.iter().enumerate() {
            if tick <= change.tick {
                break;
            }
            let next_tick = self.changes.get(i + 1).map(|c| c.tick).unwrap_or(tick);
            let span_end = next_tick.min(tick);
            let ticks = (span_end - change.tick) as f64;
            seconds += ticks * change.us_per_quarter as f64 / (self.ppq as f64 * 1_000_000.0);
        }
        seconds
    }

    /// Convert seconds back to ticks at a *constant* tempo.
    ///
    /// Used when writing new files: the writer emits one tempo event, so the
    /// inverse is exact for the files it produces.
    pub fn ticks_at_constant(seconds: f64, ppq: u16, us_per_quarter: u32) -> u64 {
        if seconds <= 0.0 {
            return 0;
        }
        let ticks = seconds * ppq as f64 * 1_000_000.0 / us_per_quarter as f64;
        ticks.round().max(0.0) as u64
    }

    /// Tempo in beats per minute active at `tick`.
    pub fn bpm_at(&self, tick: u64) -> f64 {
        let active = self
            .changes
            .iter()
            .rev()
            .find(|c| c.tick <= tick)
            .copied()
            .unwrap_or(self.changes[0]);
        60_000_000.0 / active.us_per_quarter as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tempo_is_120_bpm() {
        let map = TempoMap::new(480);
        assert!((map.bpm_at(0) - 120.0).abs() < 1e-9);
        // 480 ticks = one quarter note = 0.5 s at 120 BPM.
        assert!((map.seconds_at(480) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn tempo_change_mid_file() {
        let mut map = TempoMap::new(480);
        map.push(480, 1_000_000); // 60 BPM from beat 1
        map.finalize();
        // First beat: 0.5 s. Second beat at 60 BPM: 1.0 s.
        assert!((map.seconds_at(960) - 1.5).abs() < 1e-9);
        assert!((map.bpm_at(960) - 60.0).abs() < 1e-9);
    }

    #[test]
    fn roundtrip_seconds_to_ticks() {
        let ticks = TempoMap::ticks_at_constant(1.5, 480, DEFAULT_US_PER_QUARTER);
        assert_eq!(ticks, 1440);
    }
}
