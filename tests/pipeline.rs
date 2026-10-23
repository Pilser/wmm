//! End-to-end tests through the public API: notes → MIDI file → notes.

use std::path::PathBuf;

use wmm::harmonize::{self, Mode as HarmonyMode, Options};
use wmm::key::Key;
use wmm::{demo, midi, stats};

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join("wmm-tests");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

#[test]
fn demo_roundtrips_through_midi_and_keeps_timing() {
    let path = temp_dir().join("demo-roundtrip.mid");
    let phrase = demo::phrase();
    midi::save(&path, &phrase).expect("save");

    let loaded = midi::load(&path).expect("load");
    assert_eq!(loaded.stream.len(), phrase.len());
    assert_eq!(loaded.track_count, 1);

    for (original, parsed) in phrase.notes.iter().zip(loaded.stream.notes.iter()) {
        assert_eq!(original.pitch, parsed.pitch);
        assert_eq!(original.velocity, parsed.velocity);
        assert_eq!(original.channel, parsed.channel);
        assert!(
            (original.start - parsed.start).abs() < 1e-3,
            "start {} vs {}",
            original.start,
            parsed.start
        );
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn harmony_is_generated_in_key_and_on_its_own_channel() {
    let phrase = demo::phrase();
    let options = Options {
        mode: HarmonyMode::Third,
        key: Some(demo::key()),
        velocity_scale: 0.8,
        echo_delay: 0.35,
    };
    let response = harmonize::assign_channel(&harmonize::harmonize(&phrase, &options), 2);

    assert_eq!(response.len(), phrase.len());
    assert!(response.notes.iter().all(|n| n.channel == 2));
    assert!(response.notes.iter().all(|n| n.velocity < 96));

    // Every third above must be a member of A natural minor.
    let scale = Key::new(9, wmm::key::Mode::Minor).scale();
    for note in &response.notes {
        let pc = (note.pitch as i32 - 9).rem_euclid(12);
        assert!(
            scale.contains(&pc),
            "{} is outside A minor",
            wmm::events::note_name(note.pitch)
        );
    }

    let mut merged = phrase.clone();
    merged.extend(&response);
    assert_eq!(merged.len(), phrase.len() * 2);
}

#[test]
fn stats_see_the_whole_demo() {
    let analysis = stats::analyze(&demo::phrase());
    assert_eq!(analysis.notes, 16);
    assert!((analysis.duration_s - 4.0).abs() < 1e-9);
    assert_eq!(analysis.pitch_range(), "F2–F5");
}
