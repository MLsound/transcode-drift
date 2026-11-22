use std::f64::consts::PI;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::audio::deinterleave;

/// STFT window size in samples per channel.
pub const FFT_SIZE: usize = 2048;
/// STFT hop size in samples per channel (75% overlap).
pub const HOP_SIZE: usize = 512;
/// Relative frame power gate (-60 dB below maximum reference frame power).
const SILENCE_GATE_RATIO: f64 = 1e-6;
/// Numerical floor for power bins in Log-Spectral Distance calculation.
const LSD_EPSILON: f64 = 1e-12;

/// Time-domain Signal-to-Noise and Noise-to-Signal metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeDomainMetrics {
    pub snr_db: f64,
    pub nsr_db: f64,
    pub nsr_linear: f64,
}

/// Frequency-domain Log-Spectral Distance metrics across three evaluation bands.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralMetrics {
    pub lsd_full_db: f64,
    pub lsd_passband_db: f64,
    pub lsd_highband_db: f64,
}

/// Complete per-iteration quality evaluation metrics for a single codec chain step.
#[derive(Debug, Clone, PartialEq)]
pub struct IterationMetrics {
    pub chain: String,
    pub iteration: usize,
    pub lag_frames: isize,
    pub snr_db: f64,
    pub nsr_db: f64,
    pub nsr_linear: f64,
    pub lsd_full_db: f64,
    pub lsd_passband_db: f64,
    pub lsd_highband_db: f64,
}

/// Finds the integer frame lag `tau` in `[-max_lag_frames, +max_lag_frames]` that maximizes
/// the overlap-length-normalized cross-correlation between `reference` and `degraded`:
///
/// `R_bar(tau) = (1 / M_tau) * sum_n x_ref[n] * x_deg[n + tau * C]`
///
/// Stepping sample offsets by `channels` preserves stereo/multi-channel interleaving.
pub fn find_best_frame_lag(
    reference: &[f64],
    degraded: &[f64],
    channels: usize,
    max_lag_frames: isize,
) -> isize {
    let ch = channels.max(1);
    let ref_frames = reference.len() / ch;
    let deg_frames = degraded.len() / ch;
    let ref_len = ref_frames * ch;
    let deg_len = deg_frames * ch;

    if ref_len == 0 || deg_len == 0 {
        return 0;
    }

    let max_lag = max_lag_frames.abs();
    let mut best_lag: isize = 0;
    let mut best_corr = f64::NEG_INFINITY;

    for lag in -max_lag..=max_lag {
        let sample_shift = lag.saturating_mul(ch as isize);
        let (ref_start, deg_start) = if sample_shift >= 0 {
            (0usize, sample_shift as usize)
        } else {
            ((-sample_shift) as usize, 0usize)
        };

        if ref_start >= ref_len || deg_start >= deg_len {
            continue;
        }

        let overlap = (ref_len - ref_start).min(deg_len - deg_start);
        if overlap == 0 {
            continue;
        }

        let ref_slice = &reference[ref_start..ref_start + overlap];
        let deg_slice = &degraded[deg_start..deg_start + overlap];

        let mut dot_sum = 0.0_f64;
        for (&r, &d) in ref_slice.iter().zip(deg_slice.iter()) {
            dot_sum += r * d;
        }
        let mean_corr = dot_sum / (overlap as f64);

        if mean_corr > best_corr + 1e-15
            || ((mean_corr - best_corr).abs() <= 1e-15 && lag.abs() < best_lag.abs())
        {
            best_corr = mean_corr;
            best_lag = lag;
        }
    }

    best_lag
}

/// Slices `reference` and `degraded` to their overlapping lag-aligned region `(r_aligned, d_aligned)`.
pub fn align_signals<'a>(
    reference: &'a [f64],
    degraded: &'a [f64],
    channels: usize,
    lag_frames: isize,
) -> (&'a [f64], &'a [f64]) {
    let ch = channels.max(1);
    let ref_len = (reference.len() / ch) * ch;
    let deg_len = (degraded.len() / ch) * ch;

    let sample_shift = lag_frames.saturating_mul(ch as isize);
    let (ref_start, deg_start) = if sample_shift >= 0 {
        (0usize, sample_shift as usize)
    } else {
        ((-sample_shift) as usize, 0usize)
    };

    if ref_start >= ref_len || deg_start >= deg_len {
        return (&[], &[]);
    }

    let overlap = (ref_len - ref_start).min(deg_len - deg_start);
    (
        &reference[ref_start..ref_start + overlap],
        &degraded[deg_start..deg_start + overlap],
    )
}

/// Computes time-domain SNR (dB), NSR (dB), and linear NSR on lag-aligned slices.
pub fn compute_time_domain_metrics(r_aligned: &[f64], d_aligned: &[f64]) -> TimeDomainMetrics {
    let len = r_aligned.len().min(d_aligned.len());
    let mut p_signal = 0.0_f64;
    let mut p_noise = 0.0_f64;

    for (&x, &y) in r_aligned.iter().take(len).zip(d_aligned.iter().take(len)) {
        p_signal += x * x;
        let diff = x - y;
        p_noise += diff * diff;
    }

    if p_noise == 0.0 {
        return TimeDomainMetrics {
            snr_db: f64::INFINITY,
            nsr_db: f64::NEG_INFINITY,
            nsr_linear: 0.0,
        };
    }

    if p_signal == 0.0 {
        return TimeDomainMetrics {
            snr_db: f64::NEG_INFINITY,
            nsr_db: f64::INFINITY,
            nsr_linear: f64::INFINITY,
        };
    }

    let nsr_linear = p_noise / p_signal;
    let snr_db = 10.0 * (p_signal / p_noise).log10();
    let nsr_db = -snr_db;

    TimeDomainMetrics {
        snr_db,
        nsr_db,
        nsr_linear,
    }
}

/// Computes frequency-domain Log-Spectral Distance (LSD in dB) across:
/// - `lsd_full_db`: 20 Hz to 20 kHz
/// - `lsd_passband_db`: 20 Hz to 16 kHz
/// - `lsd_highband_db`: 16 kHz to 20 kHz
///
/// De-interleaves channels, applies a 2048-point Hann-windowed STFT with 512-sample hop,
/// gates out frames more than 60 dB below peak reference frame power, and averages LSD
/// across active frames and channels.
pub fn compute_spectral_metrics(
    r_aligned: &[f64],
    d_aligned: &[f64],
    channels: usize,
    sample_rate: u32,
) -> SpectralMetrics {
    let ch = channels.max(1);
    let r_channels = deinterleave(r_aligned, ch);
    let d_channels = deinterleave(d_aligned, ch);

    let nyquist_bin = FFT_SIZE / 2;
    let bin_for_hz = |hz: f64| -> usize {
        let raw = (hz * (FFT_SIZE as f64) / f64::from(sample_rate)).round() as usize;
        raw.clamp(1, nyquist_bin)
    };

    let full_range = (bin_for_hz(20.0), bin_for_hz(20_000.0));
    let passband_range = (bin_for_hz(20.0), bin_for_hz(16_000.0));
    let highband_range = (bin_for_hz(16_000.0), bin_for_hz(20_000.0));

    let hann_window: Vec<f64> = (0..FFT_SIZE)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * (n as f64) / ((FFT_SIZE - 1) as f64)).cos()))
        .collect();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut scratch = vec![Complex::new(0.0_f64, 0.0_f64); fft.get_inplace_scratch_len()];
    let mut fft_buf = vec![Complex::new(0.0_f64, 0.0_f64); FFT_SIZE];

    let mut sum_full = 0.0_f64;
    let mut sum_pass = 0.0_f64;
    let mut sum_high = 0.0_f64;
    let mut valid_channels = 0usize;

    for (r_mono, d_mono) in r_channels.iter().zip(d_channels.iter()) {
        let n_samples = r_mono.len().min(d_mono.len());
        if n_samples == 0 {
            continue;
        }

        let frame_starts: Vec<usize> = if n_samples < FFT_SIZE {
            vec![0]
        } else {
            (0..=(n_samples - FFT_SIZE)).step_by(HOP_SIZE).collect()
        };

        let num_frames = frame_starts.len();
        let num_bins = nyquist_bin + 1;
        let mut ref_power_frames = Vec::with_capacity(num_frames);
        let mut deg_power_frames = Vec::with_capacity(num_frames);
        let mut frame_energies = Vec::with_capacity(num_frames);
        let mut e_max = 0.0_f64;

        for &start in &frame_starts {
            // Reference STFT frame
            for i in 0..FFT_SIZE {
                let sample = if start + i < n_samples {
                    r_mono[start + i]
                } else {
                    0.0
                };
                fft_buf[i] = Complex::new(sample * hann_window[i], 0.0);
            }
            fft.process_with_scratch(&mut fft_buf, &mut scratch);

            let mut r_bins = Vec::with_capacity(num_bins);
            let mut frame_energy = 0.0_f64;
            for c in fft_buf.iter().take(num_bins) {
                let p = c.norm_sqr();
                r_bins.push(p);
                frame_energy += p;
            }
            if frame_energy > e_max {
                e_max = frame_energy;
            }
            frame_energies.push(frame_energy);
            ref_power_frames.push(r_bins);

            // Degraded STFT frame
            for i in 0..FFT_SIZE {
                let sample = if start + i < n_samples {
                    d_mono[start + i]
                } else {
                    0.0
                };
                fft_buf[i] = Complex::new(sample * hann_window[i], 0.0);
            }
            fft.process_with_scratch(&mut fft_buf, &mut scratch);

            let mut d_bins = Vec::with_capacity(num_bins);
            for c in fft_buf.iter().take(num_bins) {
                d_bins.push(c.norm_sqr());
            }
            deg_power_frames.push(d_bins);
        }

        let gate_threshold = SILENCE_GATE_RATIO * e_max;
        let mut chan_full = 0.0_f64;
        let mut chan_pass = 0.0_f64;
        let mut chan_high = 0.0_f64;
        let mut active_frames = 0usize;

        for m in 0..num_frames {
            if frame_energies[m] < gate_threshold {
                continue;
            }
            let r_bins = &ref_power_frames[m];
            let d_bins = &deg_power_frames[m];

            chan_full += frame_band_lsd(r_bins, d_bins, full_range.0, full_range.1);
            chan_pass += frame_band_lsd(r_bins, d_bins, passband_range.0, passband_range.1);
            chan_high += frame_band_lsd(r_bins, d_bins, highband_range.0, highband_range.1);
            active_frames += 1;
        }

        if active_frames > 0 {
            let denom = active_frames as f64;
            sum_full += chan_full / denom;
            sum_pass += chan_pass / denom;
            sum_high += chan_high / denom;
            valid_channels += 1;
        }
    }

    if valid_channels == 0 {
        return SpectralMetrics {
            lsd_full_db: 0.0,
            lsd_passband_db: 0.0,
            lsd_highband_db: 0.0,
        };
    }

    let ch_denom = valid_channels as f64;
    SpectralMetrics {
        lsd_full_db: sum_full / ch_denom,
        lsd_passband_db: sum_pass / ch_denom,
        lsd_highband_db: sum_high / ch_denom,
    }
}

/// Computes band-limited Log-Spectral Distance for a single STFT frame across bins `[k1, k2]`.
fn frame_band_lsd(r_bins: &[f64], d_bins: &[f64], k1: usize, k2: usize) -> f64 {
    if k2 < k1 || k2 >= r_bins.len() || k2 >= d_bins.len() {
        return 0.0;
    }
    let count = (k2 - k1 + 1) as f64;
    let mut sq_sum = 0.0_f64;

    for k in k1..=k2 {
        let ratio = (r_bins[k] + LSD_EPSILON) / (d_bins[k] + LSD_EPSILON);
        let diff_db = 10.0 * ratio.log10();
        sq_sum += diff_db * diff_db;
    }

    (sq_sum / count).sqrt()
}

/// Evaluates lag alignment, time-domain SNR/NSR, and frequency-domain LSD metrics
/// between `reference` and `degraded`.
pub fn evaluate_iteration_metrics(
    chain: &str,
    iteration: usize,
    reference: &[f64],
    degraded: &[f64],
    channels: usize,
    sample_rate: u32,
    max_lag_frames: isize,
) -> IterationMetrics {
    let lag_frames = find_best_frame_lag(reference, degraded, channels, max_lag_frames);
    let (r_aligned, d_aligned) = align_signals(reference, degraded, channels, lag_frames);
    let td = compute_time_domain_metrics(r_aligned, d_aligned);
    let sd = compute_spectral_metrics(r_aligned, d_aligned, channels, sample_rate);

    IterationMetrics {
        chain: chain.to_string(),
        iteration,
        lag_frames,
        snr_db: td.snr_db,
        nsr_db: td.nsr_db,
        nsr_linear: td.nsr_linear,
        lsd_full_db: sd.lsd_full_db,
        lsd_passband_db: sd.lsd_passband_db,
        lsd_highband_db: sd.lsd_highband_db,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::generate_synthetic_test_signal;

    #[test]
    fn test_bit_exact_identity() {
        let signal = generate_synthetic_test_signal(1.0, 48_000);
        let metrics = evaluate_iteration_metrics(
            "WAV <-> FLAC",
            1,
            &signal.samples,
            &signal.samples,
            signal.channels,
            signal.sample_rate,
            2048,
        );

        assert_eq!(metrics.lag_frames, 0);
        assert_eq!(metrics.snr_db, f64::INFINITY);
        assert_eq!(metrics.nsr_db, f64::NEG_INFINITY);
        assert_eq!(metrics.nsr_linear, 0.0);
        assert!(
            metrics.lsd_full_db < 1e-9,
            "Expected lsd_full_db < 1e-9, got {}",
            metrics.lsd_full_db
        );
        assert!(
            metrics.lsd_passband_db < 1e-9,
            "Expected lsd_passband_db < 1e-9, got {}",
            metrics.lsd_passband_db
        );
        assert!(
            metrics.lsd_highband_db < 1e-9,
            "Expected lsd_highband_db < 1e-9, got {}",
            metrics.lsd_highband_db
        );
    }

    #[test]
    fn test_priming_delay_recovery() {
        let signal = generate_synthetic_test_signal(1.0, 48_000);
        let shift_frames = 43usize;
        let shift_samples = shift_frames * signal.channels;

        // Prepend 43 stereo frames (86 samples) of priming zeros.
        let mut shifted = vec![0.0_f64; shift_samples];
        shifted.extend_from_slice(&signal.samples);

        let best_lag =
            find_best_frame_lag(&signal.samples, &shifted, signal.channels, 2048);
        assert_eq!(best_lag, 43);

        let (r_aligned, d_aligned) =
            align_signals(&signal.samples, &shifted, signal.channels, best_lag);
        let td = compute_time_domain_metrics(r_aligned, d_aligned);
        assert!(
            td.snr_db > 100.0,
            "Expected recovered SNR > 100 dB, got {}",
            td.snr_db
        );
    }

    #[test]
    fn test_band_limited_spectral_isolation() {
        let sample_rate = 48_000u32;
        let channels = 2usize;
        let duration = 1.0_f64;
        let frames = (duration * f64::from(sample_rate)) as usize;

        let mut reference = Vec::with_capacity(frames * channels);
        let mut highband_attenuated = Vec::with_capacity(frames * channels);

        // Build a broadband multi-tone signal spanning 100 Hz to 19.8 kHz.
        let tones: Vec<f64> = (1..=99).map(|k| f64::from(k) * 200.0).collect();

        for n in 0..frames {
            let t = (n as f64) / f64::from(sample_rate);
            let mut ref_l = 0.0_f64;
            let mut ref_r = 0.0_f64;
            let mut deg_l = 0.0_f64;
            let mut deg_r = 0.0_f64;

            for (idx, &freq) in tones.iter().enumerate() {
                let phase = (idx as f64) * 0.23;
                let s_l = 0.02 * (2.0 * PI * freq * t + phase).sin();
                let s_r = 0.02 * (2.0 * PI * freq * t + phase + 0.4).cos();

                ref_l += s_l;
                ref_r += s_r;

                // Attenuate frequencies above 16.5 kHz by -40 dB (factor 0.01).
                let gain = if freq > 16_500.0 { 0.01 } else { 1.0 };
                deg_l += gain * s_l;
                deg_r += gain * s_r;
            }

            reference.push(ref_l);
            reference.push(ref_r);
            highband_attenuated.push(deg_l);
            highband_attenuated.push(deg_r);
        }

        let spectral =
            compute_spectral_metrics(&reference, &highband_attenuated, channels, sample_rate);

        assert!(
            spectral.lsd_highband_db > 15.0,
            "Expected lsd_highband_db > 15.0, got {}",
            spectral.lsd_highband_db
        );
        assert!(
            spectral.lsd_passband_db < 0.5,
            "Expected lsd_passband_db < 0.5, got {}",
            spectral.lsd_passband_db
        );
    }
}
