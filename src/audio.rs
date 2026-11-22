use std::f64::consts::PI;
use std::path::Path;

use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

/// Normalized PCM audio buffer in [-1.0, 1.0] with interleaved channel layout.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub samples: Vec<f64>,
    pub channels: usize,
    pub sample_rate: u32,
}

impl AudioBuffer {
    /// Returns the number of multi-channel sample frames in the buffer.
    pub fn num_frames(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels
    }
}

/// Checks whether the reference sample rate matches libopus's native 48 kHz rate.
/// Logs a warning to stderr if internal resampling will occur.
pub fn warn_if_non_48k_sample_rate(sample_rate: u32) {
    if sample_rate != 48_000 {
        eprintln!(
            "Warning: Reference WAV sample rate is {} Hz (not 48000 Hz). \
             libopus operates natively at 48 kHz and will trigger internal resampling \
             filter ripple on every iteration.",
            sample_rate
        );
    }
}

/// Loads a 16-bit, 24-bit, 32-bit integer or 32-bit float PCM WAV file into a normalized
/// `AudioBuffer` with sample values in `[-1.0, 1.0]`.
pub fn load_wav(path: &Path) -> Result<AudioBuffer> {
    let mut reader = WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file at '{}'", path.display()))?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels);
    if channels == 0 {
        bail!("Invalid WAV file '{}': channel count is 0", path.display());
    }

    let samples: Vec<f64> = match spec.sample_format {
        SampleFormat::Float => {
            if spec.bits_per_sample != 32 {
                bail!(
                    "Unsupported float WAV bit depth {} in '{}'",
                    spec.bits_per_sample,
                    path.display()
                );
            }
            let mut out = Vec::with_capacity(reader.len() as usize);
            for sample_res in reader.samples::<f32>() {
                let s = sample_res.with_context(|| {
                    format!("Failed to decode f32 sample from '{}'", path.display())
                })?;
                out.push(f64::from(s).clamp(-1.0, 1.0));
            }
            out
        }
        SampleFormat::Int => {
            if spec.bits_per_sample == 0 || spec.bits_per_sample > 32 {
                bail!(
                    "Unsupported integer WAV bit depth {} in '{}'",
                    spec.bits_per_sample,
                    path.display()
                );
            }
            let scale = 2.0_f64.powi(i32::from(spec.bits_per_sample) - 1);
            let mut out = Vec::with_capacity(reader.len() as usize);
            for sample_res in reader.samples::<i32>() {
                let s = sample_res.with_context(|| {
                    format!("Failed to decode integer sample from '{}'", path.display())
                })?;
                let normalized = (f64::from(s) / scale).clamp(-1.0, 1.0);
                out.push(normalized);
            }
            out
        }
    };

    if samples.is_empty() {
        bail!("WAV file '{}' contains zero audio samples", path.display());
    }

    Ok(AudioBuffer {
        samples,
        channels,
        sample_rate: spec.sample_rate,
    })
}

/// Writes a normalized `AudioBuffer` to a 16-bit PCM WAV file on disk.
pub fn write_wav_16bit(path: &Path, buffer: &AudioBuffer) -> Result<()> {
    let channels_u16 = u16::try_from(buffer.channels)
        .with_context(|| format!("Invalid channel count: {}", buffer.channels))?;
    let spec = WavSpec {
        channels: channels_u16,
        sample_rate: buffer.sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("Failed to create WAV file at '{}'", path.display()))?;

    for &sample in &buffer.samples {
        let scaled = (sample * 32768.0).round();
        let quantized = scaled.clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16;
        writer
            .write_sample(quantized)
            .with_context(|| format!("Failed to write PCM sample to '{}'", path.display()))?;
    }

    writer
        .finalize()
        .with_context(|| format!("Failed to finalize WAV file '{}'", path.display()))?;
    Ok(())
}

/// Splits an interleaved audio buffer slice into per-channel mono sample vectors.
pub fn deinterleave(samples: &[f64], channels: usize) -> Vec<Vec<f64>> {
    let ch = channels.max(1);
    let frames = samples.len() / ch;
    let mut per_channel = vec![Vec::with_capacity(frames); ch];

    for frame_idx in 0..frames {
        let base = frame_idx * ch;
        for (c, chan_buf) in per_channel.iter_mut().enumerate() {
            chan_buf.push(samples[base + c]);
        }
    }

    per_channel
}

/// Generates a deterministic 3-second 48 kHz stereo synthetic test signal combining:
/// - Harmonic series across the passband (220 Hz fundamental and overtones up to 14 kHz)
/// - High-band test tones (16.8 kHz, 17.7 kHz, 18.9 kHz) to probe codec low-pass filtering
/// - Exponential frequency sweep (50 Hz to 19.5 kHz)
/// - Deterministic broadband dither/noise floor
///
/// The generated samples are quantized to 16-bit PCM steps so that round-tripping through
/// a 16-bit WAV file is bit-exact.
pub fn generate_synthetic_test_signal(duration_secs: f64, sample_rate: u32) -> AudioBuffer {
    let channels = 2usize;
    let total_frames = (duration_secs * f64::from(sample_rate)).round() as usize;
    let mut samples = Vec::with_capacity(total_frames * channels);

    let sr = f64::from(sample_rate);
    let f_start = 50.0_f64;
    let f_end = 19_500.0_f64;
    let sweep_ratio = f_end / f_start;
    let sweep_k = duration_secs / sweep_ratio.ln();

    // Deterministic xorshift64 PRNG for broadband energy across all FFT bins.
    let mut rng_state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_noise = || -> f64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let unit = (rng_state as f64) / (u64::MAX as f64);
        unit * 2.0 - 1.0
    };

    let passband_freqs: [(f64, f64); 8] = [
        (220.0, 0.16),
        (440.0, 0.13),
        (880.0, 0.10),
        (1760.0, 0.08),
        (3520.0, 0.07),
        (7040.0, 0.06),
        (11_200.0, 0.05),
        (14_400.0, 0.05),
    ];

    let highband_freqs: [(f64, f64); 3] = [
        (16_800.0, 0.09),
        (17_750.0, 0.09),
        (18_900.0, 0.08),
    ];

    for frame in 0..total_frames {
        let t = (frame as f64) / sr;

        // Smooth fade-in and fade-out envelope (20 ms) to avoid boundary click artifacts.
        let fade_time = 0.02;
        let env = if t < fade_time {
            0.5 * (1.0 - (PI * t / fade_time).cos())
        } else if t > (duration_secs - fade_time) {
            let rem = (duration_secs - t).max(0.0);
            0.5 * (1.0 - (PI * rem / fade_time).cos())
        } else {
            1.0
        };

        let mut left = 0.0_f64;
        let mut right = 0.0_f64;

        for (idx, &(freq, amp)) in passband_freqs.iter().enumerate() {
            let phase_offset = (idx as f64) * 0.37;
            left += amp * (2.0 * PI * freq * t + phase_offset).sin();
            right += amp * (2.0 * PI * freq * t + phase_offset + 0.45).cos();
        }

        for (idx, &(freq, amp)) in highband_freqs.iter().enumerate() {
            let phase_offset = (idx as f64) * 0.61;
            left += amp * (2.0 * PI * freq * t + phase_offset).sin();
            right += amp * (2.0 * PI * freq * t + phase_offset + 0.25).sin();
        }

        // Exponential chirp from 50 Hz to 19.5 kHz.
        let sweep_phase = 2.0 * PI * f_start * sweep_k * ((t / sweep_k).exp() - 1.0);
        left += 0.10 * sweep_phase.sin();
        right += 0.10 * (sweep_phase + 0.3).cos();

        // Broadband noise floor.
        left += 0.04 * next_noise();
        right += 0.04 * next_noise();

        // Apply envelope and quantize to exact 16-bit PCM grid.
        for val in [left * env * 0.85, right * env * 0.85] {
            let clamped = val.clamp(-0.98, 0.98);
            let q = ((clamped * 32768.0).round())
                .clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16;
            samples.push(f64::from(q) / 32768.0);
        }
    }

    AudioBuffer {
        samples,
        channels,
        sample_rate,
    }
}
