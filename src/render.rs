//! Render a MIDI file to WAV using a SoundFont (`.sf2`), in pure Rust.
//!
//! A SoundFont is a file of recorded instrument samples plus mapping rules: it
//! tells the synthesizer which sample to play for each note. We never generate
//! timbre ourselves; the SoundFont (or later, a DAW/VST) does that.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rustysynth::{MidiFile, MidiFileSequencer, SoundFont, Synthesizer, SynthesizerSettings};

/// Options for rendering.
#[derive(Debug, Clone, Copy)]
pub struct RenderOptions {
    /// Sample rate in hertz (audio samples per second).
    pub sample_rate: u32,
    /// Linear gain applied to the mix. 1.0 is the synthesizer's own level.
    pub gain: f32,
    /// Enable the SoundFont's built-in reverb and chorus.
    pub effects: bool,
    /// Extra seconds of silence appended so reverb tails are not cut off.
    pub tail_s: f64,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            gain: 1.0,
            effects: true,
            tail_s: 1.0,
        }
    }
}

/// What was rendered, for logging.
#[derive(Debug, Clone, Copy)]
pub struct RenderReport {
    pub frames: u64,
    pub seconds: f64,
    pub sample_rate: u32,
}

/// Render `midi_path` to `wav_path` with the SoundFont at `sf2_path`.
pub fn render_midi_to_wav(
    midi_path: &Path,
    sf2_path: &Path,
    wav_path: &Path,
    options: &RenderOptions,
) -> Result<RenderReport> {
    if !sf2_path.exists() {
        bail!(
            "SoundFont not found: {} (try /usr/share/sounds/sf2/TimGM6mb.sf2)",
            sf2_path.display()
        );
    }
    let sample_rate = options.sample_rate.clamp(16_000, 192_000) as i32;

    let mut sf2_file = File::open(sf2_path)
        .with_context(|| format!("cannot open SoundFont {}", sf2_path.display()))?;
    let sound_font =
        Arc::new(SoundFont::new(&mut sf2_file).context("cannot parse SoundFont (is it .sf2?)")?);

    let mut midi_file = File::open(midi_path)
        .with_context(|| format!("cannot open MIDI file {}", midi_path.display()))?;
    let midi = Arc::new(MidiFile::new(&mut midi_file).context("cannot parse MIDI file")?);

    let mut settings = SynthesizerSettings::new(sample_rate);
    settings.enable_reverb_and_chorus = options.effects;
    let synthesizer = Synthesizer::new(&sound_font, &settings)
        .context("cannot create synthesizer from SoundFont")?;
    let mut sequencer = MidiFileSequencer::new(synthesizer);
    sequencer.play(&midi, false);

    let block = settings.block_size;
    let mut left = vec![0f32; block];
    let mut right = vec![0f32; block];

    if let Some(parent) = wav_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("cannot create {}", parent.display()))?;
        }
    }
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: options.sample_rate.clamp(16_000, 192_000),
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let file =
        File::create(wav_path).with_context(|| format!("cannot create {}", wav_path.display()))?;
    let mut writer =
        hound::WavWriter::new(BufWriter::new(file), spec).context("cannot create WAV writer")?;

    let gain = options.gain.max(0.0);
    let mut frames = 0u64;

    while !sequencer.end_of_sequence() {
        sequencer.render(&mut left, &mut right);
        for i in 0..block {
            let l = to_i16(left[i] * gain);
            let r = to_i16(right[i] * gain);
            writer.write_sample(l).context("cannot write WAV sample")?;
            writer.write_sample(r).context("cannot write WAV sample")?;
        }
        frames += block as u64;
    }

    // Silence tail for reverb / release. Use the last block's content as a hint
    // that the sequencer is done; the synthesizer keeps ringing down naturally.
    let tail_frames = (options.tail_s.max(0.0) * sample_rate as f64) as u64;
    let mut silence_left = 0u64;
    while silence_left < tail_frames {
        sequencer.render(&mut left, &mut right);
        for i in 0..block {
            let l = to_i16(left[i] * gain);
            let r = to_i16(right[i] * gain);
            writer.write_sample(l).context("cannot write WAV sample")?;
            writer.write_sample(r).context("cannot write WAV sample")?;
        }
        frames += block as u64;
        silence_left += block as u64;
    }

    writer.finalize().context("cannot finalize WAV file")?;

    Ok(RenderReport {
        frames,
        seconds: frames as f64 / sample_rate as f64,
        sample_rate: options.sample_rate,
    })
}

fn to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamping_never_wraps() {
        assert_eq!(to_i16(2.0), i16::MAX);
        assert_eq!(to_i16(-2.0), -i16::MAX);
        assert_eq!(to_i16(0.0), 0);
    }
}
