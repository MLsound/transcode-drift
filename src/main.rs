pub mod audio;
pub mod cli;
pub mod metrics;
pub mod report;
pub mod transcode;

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;

use crate::audio::{
    generate_synthetic_test_signal, load_wav, warn_if_non_48k_sample_rate,
};
use crate::cli::CliArgs;
use crate::report::{
    compute_average_metrics, print_batch_summary, print_console_summary,
    trigger_history_plot, write_csv_report, FileBatchRecord,
};
use crate::transcode::{run_benchmark_chains, verify_ffmpeg_installed, TranscodeConfig};

/// Discovers all `.wav` audio files inside a directory, sorted alphabetically by path.
pub fn discover_wav_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let entries = fs::read_dir(dir)
        .with_context(|| format!("Failed to read input directory '{}'", dir.display()))?;

    let mut wav_files = Vec::new();
    for entry_res in entries {
        let entry = entry_res
            .with_context(|| format!("Failed reading entry in '{}'", dir.display()))?;
        let path = entry.path();
        if path.is_file() {
            let is_wav = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false);
            if is_wav {
                wav_files.push(path);
            }
        }
    }

    wav_files.sort();
    Ok(wav_files)
}

/// Displays the discovered audio file count and names, then asks the user for confirmation.
fn confirm_batch_execution(dir: &Path, files: &[PathBuf]) -> Result<bool> {
    println!(
        "Discovered {} WAV audio file(s) in '{}':",
        files.len(),
        dir.display()
    );
    for (idx, file_path) in files.iter().enumerate() {
        let name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unknown>");
        println!("  {}. {}", idx + 1, name);
    }

    print!(
        "\nProceed with batch conversion for all {} file(s)? [y/N]: ",
        files.len()
    );
    io::stdout()
        .flush()
        .context("Failed to flush stdout before reading user confirmation")?;

    let mut response = String::new();
    io::stdin()
        .read_line(&mut response)
        .context("Failed to read confirmation response from stdin")?;

    Ok(is_affirmative_confirmation(&response))
}

/// Evaluates whether a raw user confirmation string is affirmative (`y` or `yes`).
pub fn is_affirmative_confirmation(input: &str) -> bool {
    let trimmed = input.trim();
    trimmed.eq_ignore_ascii_case("y") || trimmed.eq_ignore_ascii_case("yes")
}

fn main() -> Result<()> {
    let args = CliArgs::parse();

    // 1. Verify ffmpeg binary availability on PATH.
    verify_ffmpeg_installed()?;

    // 2. Check if --input points to a directory for batch testing.
    if let Some(input_path) = &args.input {
        if input_path.is_dir() {
            let wav_files = discover_wav_files(input_path)?;
            if wav_files.is_empty() {
                bail!(
                    "No .wav audio files found in input folder '{}'",
                    input_path.display()
                );
            }

            if !confirm_batch_execution(input_path, &wav_files)? {
                println!("Batch conversion cancelled by user.");
                return Ok(());
            }

            let batch_start = Instant::now();
            let total_files = wav_files.len();
            let mut batch_records = Vec::with_capacity(total_files);

            for (idx, file_path) in wav_files.iter().enumerate() {
                let file_name = file_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("audio.wav")
                    .to_string();

                println!(
                    "\n[{}/{}] Loading '{}'...",
                    idx + 1,
                    total_files,
                    file_path.display()
                );
                let reference_audio = load_wav(file_path)?;
                warn_if_non_48k_sample_rate(reference_audio.sample_rate);
                println!(
                    "Loaded '{}' ({} Hz, {} ch, {} frames)",
                    file_name,
                    reference_audio.sample_rate,
                    reference_audio.channels,
                    reference_audio.num_frames()
                );

                let file_start = Instant::now();
                let config = TranscodeConfig {
                    chain: args.chain,
                    iterations: args.iterations,
                    mp3_kbps: args.mp3_kbps,
                    opus_kbps: args.opus_kbps,
                    max_lag_frames: args.max_lag_frames,
                    keep_temp: args.keep_temp.clone(),
                    source_label: file_name.clone(),
                };

                let file_metrics =
                    run_benchmark_chains(Some(file_path), &reference_audio, &config)?;
                let file_elapsed = file_start.elapsed().as_secs_f64();

                print_console_summary(&file_metrics);
                println!(
                    "File [{}/{}] '{}' finished in {:.3}s",
                    idx + 1,
                    total_files,
                    file_name,
                    file_elapsed
                );

                batch_records.push(FileBatchRecord {
                    file_name,
                    sample_rate: reference_audio.sample_rate,
                    channels: reference_audio.channels,
                    num_frames: reference_audio.num_frames(),
                    elapsed_secs: file_elapsed,
                    metrics: file_metrics,
                });
            }

            let total_batch_secs = batch_start.elapsed().as_secs_f64();
            let averaged_metrics = compute_average_metrics(&batch_records);

            print_batch_summary(&batch_records, &averaged_metrics, total_batch_secs);

            write_csv_report(&args.csv_out, &averaged_metrics)?;
            println!(
                "Exported batch-averaged CSV degradation report to '{}'",
                args.csv_out.display()
            );

            if let Some(plot_out) = &args.plot_out {
                trigger_history_plot(&args.csv_out, plot_out);
            }

            return Ok(());
        }
    }

    // 3. Single-file or built-in synthetic signal execution.
    let (reference_audio, source_label) = match &args.input {
        Some(path) => {
            let buf = load_wav(path)?;
            warn_if_non_48k_sample_rate(buf.sample_rate);
            let label = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("input.wav")
                .to_string();
            println!(
                "Loaded reference WAV '{}' ({} Hz, {} ch, {} frames)",
                path.display(),
                buf.sample_rate,
                buf.channels,
                buf.num_frames()
            );
            (buf, label)
        }
        None => {
            let buf = generate_synthetic_test_signal(3.0, 48_000);
            println!(
                "Generated built-in 3.0s synthetic reference signal ({} Hz, {} ch, {} frames)",
                buf.sample_rate,
                buf.channels,
                buf.num_frames()
            );
            (buf, "synthetic_48k.wav".to_string())
        }
    };

    let run_start = Instant::now();
    let config = TranscodeConfig {
        chain: args.chain,
        iterations: args.iterations,
        mp3_kbps: args.mp3_kbps,
        opus_kbps: args.opus_kbps,
        max_lag_frames: args.max_lag_frames,
        keep_temp: args.keep_temp.clone(),
        source_label: source_label.clone(),
    };

    let metrics = run_benchmark_chains(args.input.as_deref(), &reference_audio, &config)?;
    let run_elapsed = run_start.elapsed().as_secs_f64();

    print_console_summary(&metrics);
    println!("Total execution runtime for '{}': {:.3}s", source_label, run_elapsed);

    write_csv_report(&args.csv_out, &metrics)?;
    println!("Exported CSV degradation report to '{}'", args.csv_out.display());

    if let Some(plot_out) = &args.plot_out {
        trigger_history_plot(&args.csv_out, plot_out);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::write_wav_16bit;
    use tempfile::TempDir;

    #[test]
    fn test_discover_wav_files_and_confirmation_parsing() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory for batch test");
        let signal = generate_synthetic_test_signal(0.1, 48_000);

        let wav_b = temp_dir.path().join("track_b.WAV");
        let wav_a = temp_dir.path().join("track_a.wav");
        let non_wav = temp_dir.path().join("notes.txt");

        write_wav_16bit(&wav_b, &signal).expect("Failed to write track_b.WAV");
        write_wav_16bit(&wav_a, &signal).expect("Failed to write track_a.wav");
        fs::write(&non_wav, b"ignored file").expect("Failed to write notes.txt");

        let discovered =
            discover_wav_files(temp_dir.path()).expect("Directory discovery should succeed");
        assert_eq!(discovered.len(), 2);
        assert_eq!(discovered[0], wav_a);
        assert_eq!(discovered[1], wav_b);

        assert!(is_affirmative_confirmation("y\n"));
        assert!(is_affirmative_confirmation("YES"));
        assert!(!is_affirmative_confirmation("n"));
        assert!(!is_affirmative_confirmation(""));
    }
}

