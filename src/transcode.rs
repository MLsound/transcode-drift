use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use tempfile::TempDir;

use crate::audio::{load_wav, write_wav_16bit, AudioBuffer};
use crate::cli::ChainSelection;
use crate::metrics::{evaluate_iteration_metrics, IterationMetrics};

pub const CHAIN_LOSSLESS: &str = "WAV <-> FLAC";
pub const CHAIN_LOSSY: &str = "MP3 <-> Opus";

/// Configuration for an iterative transcoding benchmark run.
#[derive(Debug, Clone)]
pub struct TranscodeConfig {
    pub chain: ChainSelection,
    pub iterations: usize,
    pub mp3_kbps: u32,
    pub opus_kbps: u32,
    pub max_lag_frames: isize,
    pub keep_temp: Option<PathBuf>,
    pub source_label: String,
}

/// Verifies that the external `ffmpeg` binary is installed and available on `PATH`.
pub fn verify_ffmpeg_installed() -> Result<()> {
    let output = Command::new("ffmpeg").arg("-version").output();
    match output {
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            bail!(
                "ffmpeg -version failed with status {:?}: {}\n\
                 Ensure a working ffmpeg installation is available on PATH.",
                out.status.code(),
                stderr.trim()
            )
        }
        Err(err) => {
            bail!(
                "Failed to execute 'ffmpeg -version' ({}).\n\
                 Please install ffmpeg and ensure it is on your PATH \
                 (e.g., 'brew install ffmpeg' on macOS or 'sudo apt install ffmpeg' on Linux).",
                err
            )
        }
    }
}

/// Executes `ffmpeg` with `-y -loglevel error` prepended, capturing `stderr` on failure.
pub fn run_ffmpeg<I, S>(args: I) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y").arg("-loglevel").arg("error");
    for arg in args {
        cmd.arg(arg);
    }

    let output = cmd
        .output()
        .context("Failed to spawn ffmpeg process")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "ffmpeg exited with status {:?}: {}",
            output.status.code(),
            stderr.trim()
        );
    }

    Ok(())
}

/// Manages either an RAII temporary directory or a user-specified persistent directory.
enum ScratchWorkspace {
    Temp(TempDir),
    Persistent(PathBuf),
}

impl ScratchWorkspace {
    fn new(keep_temp: Option<&Path>) -> Result<Self> {
        match keep_temp {
            Some(dir) => {
                fs::create_dir_all(dir).with_context(|| {
                    format!(
                        "Failed to create intermediate scratch directory '{}'",
                        dir.display()
                    )
                })?;
                Ok(Self::Persistent(dir.to_path_buf()))
            }
            None => {
                let temp_dir = TempDir::new()
                    .context("Failed to create temporary directory for transcoding")?;
                Ok(Self::Temp(temp_dir))
            }
        }
    }

    fn path(&self) -> &Path {
        match self {
            Self::Temp(dir) => dir.path(),
            Self::Persistent(path) => path.as_path(),
        }
    }
}

/// Runs the selected iterative codec chain(s) (`WAV <-> FLAC` and/or `MP3 <-> Opus`)
/// for `N` iterations and returns all collected per-iteration degradation metrics.
pub fn run_benchmark_chains(
    reference_wav_path: Option<&Path>,
    reference_audio: &AudioBuffer,
    config: &TranscodeConfig,
) -> Result<Vec<IterationMetrics>> {
    if config.iterations == 0 {
        bail!("--iterations must be at least 1");
    }

    let workspace = ScratchWorkspace::new(config.keep_temp.as_deref())?;
    let work_dir = workspace.path();

    // Ensure a reference WAV exists on disk for ffmpeg to consume.
    let ref_wav_on_disk: PathBuf = match reference_wav_path {
        Some(p) => p.to_path_buf(),
        None => {
            let synth_path = work_dir.join("reference_synthetic_48k.wav");
            write_wav_16bit(&synth_path, reference_audio)?;
            synth_path
        }
    };

    let chain_multiplier = match config.chain {
        ChainSelection::Both => 2,
        ChainSelection::Lossless | ChainSelection::Lossy => 1,
    };
    let mut results = Vec::with_capacity(config.iterations * chain_multiplier);

    // Chain 1: Lossless baseline (WAV <-> FLAC)
    if matches!(config.chain, ChainSelection::Both | ChainSelection::Lossless) {
        let flac_results =
            run_wav_flac_chain(&ref_wav_on_disk, reference_audio, work_dir, config)?;
        results.extend(flac_results);
    }

    // Chain 2: Lossy web-transfer chain (MP3 <-> Opus)
    if matches!(config.chain, ChainSelection::Both | ChainSelection::Lossy) {
        let lossy_results =
            run_mp3_opus_chain(&ref_wav_on_disk, reference_audio, work_dir, config)?;
        results.extend(lossy_results);
    }

    Ok(results)
}

/// Executes `N` round-trip iterations of `WAV -> FLAC -> WAV (pcm_s16le)`.
fn run_wav_flac_chain(
    initial_wav: &Path,
    reference_audio: &AudioBuffer,
    work_dir: &Path,
    config: &TranscodeConfig,
) -> Result<Vec<IterationMetrics>> {
    let mut metrics_list = Vec::with_capacity(config.iterations);
    let mut current_wav = initial_wav.to_path_buf();

    for i in 1..=config.iterations {
        let iter_start = Instant::now();
        let flac_path = work_dir.join(format!("chain1_iter_{:03}.flac", i));
        let decoded_wav_path = work_dir.join(format!("chain1_iter_{:03}.wav", i));

        run_ffmpeg([
            OsStr::new("-i"),
            current_wav.as_os_str(),
            OsStr::new("-c:a"),
            OsStr::new("flac"),
            flac_path.as_os_str(),
        ])
        .with_context(|| {
            format!(
                "Chain '{}' failed encoding FLAC at iteration {}",
                CHAIN_LOSSLESS, i
            )
        })?;

        run_ffmpeg([
            OsStr::new("-i"),
            flac_path.as_os_str(),
            OsStr::new("-c:a"),
            OsStr::new("pcm_s16le"),
            decoded_wav_path.as_os_str(),
        ])
        .with_context(|| {
            format!(
                "Chain '{}' failed decoding FLAC to WAV at iteration {}",
                CHAIN_LOSSLESS, i
            )
        })?;

        let degraded = load_wav(&decoded_wav_path)?;
        let iter_metrics = evaluate_iteration_metrics(
            CHAIN_LOSSLESS,
            i,
            &reference_audio.samples,
            &degraded.samples,
            reference_audio.channels,
            reference_audio.sample_rate,
            config.max_lag_frames,
        );
        metrics_list.push(iter_metrics);

        println!(
            "  [{}] [{}] Iteration {}/{} runtime: {:.3}s",
            config.source_label,
            CHAIN_LOSSLESS,
            i,
            config.iterations,
            iter_start.elapsed().as_secs_f64()
        );

        current_wav = decoded_wav_path;
    }

    Ok(metrics_list)
}

/// Performs an initial reference WAV -> MP3 encode, then executes `N` iterations of
/// `MP3 -> Opus -> MP3` followed by decoding that iteration's MP3 to `pcm_s16le` WAV.
fn run_mp3_opus_chain(
    initial_wav: &Path,
    reference_audio: &AudioBuffer,
    work_dir: &Path,
    config: &TranscodeConfig,
) -> Result<Vec<IterationMetrics>> {
    let mut metrics_list = Vec::with_capacity(config.iterations);
    let mp3_bitrate = format!("{}k", config.mp3_kbps);
    let opus_bitrate = format!("{}k", config.opus_kbps);

    let initial_mp3_path = work_dir.join("chain2_iter_000_seed.mp3");
    run_ffmpeg([
        OsStr::new("-i"),
        initial_wav.as_os_str(),
        OsStr::new("-c:a"),
        OsStr::new("libmp3lame"),
        OsStr::new("-b:a"),
        OsStr::new(&mp3_bitrate),
        initial_mp3_path.as_os_str(),
    ])
    .with_context(|| format!("Chain '{}' failed initial MP3 encode", CHAIN_LOSSY))?;

    let mut current_mp3 = initial_mp3_path;

    for i in 1..=config.iterations {
        let iter_start = Instant::now();
        let opus_path = work_dir.join(format!("chain2_iter_{:03}.opus", i));
        let next_mp3_path = work_dir.join(format!("chain2_iter_{:03}.mp3", i));
        let decoded_wav_path = work_dir.join(format!("chain2_iter_{:03}_eval.wav", i));

        // Step 1: MP3 -> Opus
        run_ffmpeg([
            OsStr::new("-i"),
            current_mp3.as_os_str(),
            OsStr::new("-c:a"),
            OsStr::new("libopus"),
            OsStr::new("-b:a"),
            OsStr::new(&opus_bitrate),
            opus_path.as_os_str(),
        ])
        .with_context(|| {
            format!(
                "Chain '{}' failed MP3 -> Opus at iteration {}",
                CHAIN_LOSSY, i
            )
        })?;

        // Step 2: Opus -> MP3
        run_ffmpeg([
            OsStr::new("-i"),
            opus_path.as_os_str(),
            OsStr::new("-c:a"),
            OsStr::new("libmp3lame"),
            OsStr::new("-b:a"),
            OsStr::new(&mp3_bitrate),
            next_mp3_path.as_os_str(),
        ])
        .with_context(|| {
            format!(
                "Chain '{}' failed Opus -> MP3 at iteration {}",
                CHAIN_LOSSY, i
            )
        })?;

        // Step 3: Decode iteration MP3 -> 16-bit PCM WAV for evaluation
        run_ffmpeg([
            OsStr::new("-i"),
            next_mp3_path.as_os_str(),
            OsStr::new("-c:a"),
            OsStr::new("pcm_s16le"),
            decoded_wav_path.as_os_str(),
        ])
        .with_context(|| {
            format!(
                "Chain '{}' failed decoding MP3 to WAV at iteration {}",
                CHAIN_LOSSY, i
            )
        })?;

        let degraded = load_wav(&decoded_wav_path)?;
        let iter_metrics = evaluate_iteration_metrics(
            CHAIN_LOSSY,
            i,
            &reference_audio.samples,
            &degraded.samples,
            reference_audio.channels,
            reference_audio.sample_rate,
            config.max_lag_frames,
        );
        metrics_list.push(iter_metrics);

        println!(
            "  [{}] [{}] Iteration {}/{} runtime: {:.3}s",
            config.source_label,
            CHAIN_LOSSY,
            i,
            config.iterations,
            iter_start.elapsed().as_secs_f64()
        );

        current_mp3 = next_mp3_path;
    }

    Ok(metrics_list)
}

