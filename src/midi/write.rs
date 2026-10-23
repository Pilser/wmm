//! Write a [`NoteStream`] to a Standard MIDI File.
//!
//! Output is a single-track (format 0) file at a fixed 120 BPM tempo. That is
//! enough for auditioning responses: the important data is pitch, velocity and
//! relative timing.

use std::path::Path;

use anyhow::{Context, Result};
use midly::num::{u15, u24, u28, u4, u7};
use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind};

use crate::events::NoteStream;
use crate::midi::tempo::{TempoMap, DEFAULT_US_PER_QUARTER};

/// Resolution used for files we write: 480 ticks per quarter note.
pub const DEFAULT_PPQ: u16 = 480;

/// Build an in-memory Standard MIDI File from a note stream.
pub fn build_smf(stream: &NoteStream, ppq: u16) -> Result<Smf<'static>> {
    let ppq = u15::try_from(ppq.max(1)).context("ppq out of range")?;

    // (tick, priority, event); priority 0 = note-off sorts before note-on.
    let mut events: Vec<(u64, u8, TrackEvent<'static>)> = Vec::with_capacity(stream.len() * 2);
    for note in &stream.notes {
        let start = TempoMap::ticks_at_constant(note.start, ppq.as_int(), DEFAULT_US_PER_QUARTER);
        let end = TempoMap::ticks_at_constant(note.end(), ppq.as_int(), DEFAULT_US_PER_QUARTER)
            .max(start + 1);
        let channel = u4::try_from(note.channel.min(15)).context("channel out of range")?;
        let key = u7::try_from(note.pitch.min(127)).context("pitch out of range")?;
        let vel = u7::try_from(note.velocity.clamp(1, 127)).context("velocity out of range")?;
        events.push((
            start,
            1,
            TrackEvent {
                delta: u28::new(0),
                kind: TrackEventKind::Midi {
                    channel,
                    message: MidiMessage::NoteOn { key, vel },
                },
            },
        ));
        events.push((
            end,
            0,
            TrackEvent {
                delta: u28::new(0),
                kind: TrackEventKind::Midi {
                    channel,
                    message: MidiMessage::NoteOff {
                        key,
                        vel: u7::new(0),
                    },
                },
            },
        ));
    }
    events.sort_by_key(|(tick, priority, _)| (*tick, *priority));

    let mut track = Vec::with_capacity(events.len() + 2);
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(
            u24::try_from(DEFAULT_US_PER_QUARTER).expect("default tempo fits in 24 bits"),
        )),
    });
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::TimeSignature(4, 2, 24, 8)),
    });

    let mut previous = 0u64;
    for (tick, _, mut event) in events {
        // Delta times are 28-bit variable-length integers.
        let delta = tick.saturating_sub(previous).min(0x0FFF_FFFF);
        event.delta = u28::new(delta as u32);
        previous = tick;
        track.push(event);
    }
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });

    let header = Header::new(Format::SingleTrack, Timing::Metrical(ppq));
    Ok(Smf {
        header,
        tracks: vec![track],
    })
}

/// Serialize a file to bytes.
pub fn to_bytes(smf: &Smf<'_>) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    smf.write_std(&mut bytes)
        .context("cannot serialize MIDI file")?;
    Ok(bytes)
}

/// Write a note stream to `path`.
pub fn save(path: &Path, stream: &NoteStream) -> Result<()> {
    let bytes = to_bytes(&build_smf(stream, DEFAULT_PPQ)?)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("cannot create {}", parent.display()))?;
        }
    }
    std::fs::write(path, bytes).with_context(|| format!("cannot write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{Note, NoteStream};

    #[test]
    fn empty_stream_is_a_valid_file() {
        let smf = build_smf(&NoteStream::new(), 480).expect("build");
        let bytes = to_bytes(&smf).expect("serialize");
        assert!(bytes.starts_with(b"MThd"));
        let parsed = Smf::parse(&bytes).expect("reparse");
        assert_eq!(parsed.tracks.len(), 1);
    }

    #[test]
    fn note_offs_come_before_note_ons_at_same_tick() {
        let stream = NoteStream::from_notes(vec![
            Note::new(0.0, 0.5, 60, 100, 0),
            Note::new(0.5, 0.5, 62, 100, 0),
        ]);
        let smf = build_smf(&stream, 480).expect("build");
        let mut kinds = Vec::new();
        for event in &smf.tracks[0] {
            if let TrackEventKind::Midi { message, .. } = event.kind {
                kinds.push(message);
            }
        }
        // 0: on 60, 1: off 60, 2: on 62, 3: off 62
        assert!(matches!(kinds[0], MidiMessage::NoteOn { .. }));
        assert!(matches!(kinds[1], MidiMessage::NoteOff { .. }));
    }
}
