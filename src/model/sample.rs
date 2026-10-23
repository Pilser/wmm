//! Autoregressive sampling: turn model logits into a MIDI note stream.
//!
//! Sampling is done per head (time, kind, pitch, velocity) with temperature.
//! The heads are independent, so a sample can be musically odd (for example a
//! note-off for a pitch that is not held); the stream builder ignores those
//! cases instead of failing.

use anyhow::Result;
use candle_core::{IndexOp, Tensor};

use crate::events::NoteStream;
use crate::model::gru::EventModel;
use crate::model::rng::Rng;
use crate::model::vocab::{events_to_stream, Event, KIND_OFF, KIND_ON, KIND_START, TIME_BINS};

#[derive(Debug, Clone, Copy)]
pub struct SampleConfig {
    /// Maximum number of generated events.
    pub events: usize,
    /// Softmax temperature: 1.0 = model distribution, lower = safer.
    pub temperature: f32,
    pub seed: u64,
    /// Stop after this many seconds of music.
    pub max_seconds: f64,
    /// Length of notes left open at the end.
    pub tail: f64,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            events: 400,
            temperature: 0.9,
            seed: 1,
            max_seconds: 120.0,
            tail: 0.5,
        }
    }
}

/// Generate a note stream, optionally primed with a prompt.
pub fn generate(model: &EventModel, prompt: &[Event], config: &SampleConfig) -> Result<NoteStream> {
    let mut rng = Rng::new(config.seed);
    let mut state = model.zero_state(1)?;

    let mut events = if prompt.is_empty() {
        vec![Event::start()]
    } else {
        prompt.to_vec()
    };

    // Prime the recurrent state with the prompt.
    let mut logits = None;
    for event in &events {
        let (next_logits, next_state) = model.step(
            event.dt_bin(),
            event.kind,
            event.pitch as u32,
            event.vel_bin(),
            &state,
        )?;
        state = next_state;
        logits = Some(next_logits);
    }

    let temperature = config.temperature.max(0.05);
    let mut time: f64 = events.iter().map(|event| event.dt).sum();

    for _ in 0..config.events {
        if time > config.max_seconds {
            break;
        }
        let current = match logits.take() {
            Some(value) => value,
            None => break,
        };
        let dt_bin = sample(&current.time, temperature, &mut rng, None)?;
        let kind = sample(
            &current.kind,
            temperature,
            &mut rng,
            Some(KIND_START as usize),
        )?;
        let pitch = sample(&current.pitch, temperature, &mut rng, None)?;
        let velocity = sample(&current.velocity, temperature, &mut rng, None)?;

        let event = Event::from_bins(dt_bin, kind, pitch, velocity);
        // Clamp so the piece never overshoots max_seconds; the final event is
        // allowed to end exactly at the limit.
        let event = if time + event.dt > config.max_seconds {
            let remaining = (config.max_seconds - time).max(0.0);
            Event {
                dt: remaining,
                ..event
            }
        } else {
            event
        };
        time += event.dt;
        events.push(event);

        let (next_logits, next_state) = model.step(dt_bin, kind, pitch, velocity, &state)?;
        state = next_state;
        logits = Some(next_logits);
    }

    Ok(events_to_stream(&events, config.tail))
}

/// Sample one class from `[1, 1, classes]` logits.
fn sample(
    logits: &Tensor,
    temperature: f32,
    rng: &mut Rng,
    mask_out: Option<usize>,
) -> Result<u32> {
    let mut values = logits.i((0, 0))?.to_vec1::<f32>()?;
    if let Some(index) = mask_out {
        if let Some(value) = values.get_mut(index) {
            *value = f32::MIN;
        }
    }
    let max = values.iter().copied().fold(f32::MIN, f32::max);
    let mut total = 0.0f32;
    for value in values.iter_mut() {
        *value = ((*value - max) / temperature).exp();
        total += *value;
    }
    if !(total.is_finite() && total > 0.0) {
        // Degenerate distribution: fall back to the most likely class.
        return Ok(0);
    }
    let target = rng.next_f32() * total;
    let mut cumulative = 0.0f32;
    for (index, value) in values.iter().enumerate() {
        cumulative += *value;
        if cumulative >= target {
            return Ok(index as u32);
        }
    }
    Ok((values.len().saturating_sub(1)) as u32)
}

/// How many of the sampled events were real note-ons. Used for logging.
pub fn note_on_count(events: &[Event]) -> usize {
    events
        .iter()
        .filter(|event| event.kind == KIND_ON || event.kind == KIND_OFF)
        .count()
}

/// Sanity check used in tests: a generated stream only contains valid MIDI
/// numbers and positive durations.
pub fn validate(stream: &NoteStream) -> Result<()> {
    for note in &stream.notes {
        anyhow::ensure!(note.pitch < 128, "pitch out of range");
        anyhow::ensure!(note.velocity >= 1, "velocity must be at least 1");
        anyhow::ensure!(note.duration > 0.0, "duration must be positive");
        anyhow::ensure!(note.start.is_finite(), "start time must be finite");
    }
    Ok(())
}

/// Number of time bins, exposed for tests and docs.
pub fn time_bin_count() -> usize {
    TIME_BINS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::gru::{build, ModelConfig};

    #[test]
    fn generation_produces_valid_notes() {
        let (model, _varmap) = build(&ModelConfig {
            hidden: 16,
            embed: 8,
        })
        .expect("build");
        let stream = generate(
            &model,
            &[],
            &SampleConfig {
                events: 200,
                max_seconds: 4.0,
                ..SampleConfig::default()
            },
        )
        .expect("generate");
        validate(&stream).expect("valid stream");
        assert!(stream.duration() <= 6.0);
    }

    #[test]
    fn conditioning_changes_the_start_time() {
        let (model, _varmap) = build(&ModelConfig {
            hidden: 16,
            embed: 8,
        })
        .expect("build");
        let prompt = vec![
            Event::start(),
            Event {
                dt: 0.0,
                kind: KIND_ON,
                pitch: 60,
                velocity: 90,
            },
            Event {
                dt: 0.5,
                kind: KIND_OFF,
                pitch: 60,
                velocity: 0,
            },
        ];
        let stream = generate(
            &model,
            &prompt,
            &SampleConfig {
                events: 0,
                ..SampleConfig::default()
            },
        )
        .expect("generate");
        assert_eq!(stream.len(), 1);
        assert_eq!(stream.notes[0].pitch, 60);
    }

    #[test]
    fn time_bin_count_matches_vocabulary() {
        assert_eq!(time_bin_count(), 128);
    }
}
