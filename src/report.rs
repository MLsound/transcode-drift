use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};

use crate::metrics::IterationMetrics;

/// Execution record for a single audio file in a batch benchmark run.
#[derive(Debug, Clone)]
pub struct FileBatchRecord {
    pub file_name: String,
    pub sample_rate: u32,
    pub channels: usize,
    pub num_frames: usize,
    pub elapsed_secs: f64,
    pub metrics: Vec<IterationMetrics>,
}

/// Formats floating-point numbers cleanly, rendering `+inf` as `inf` and `-inf` as `-inf`.
pub fn format_float(val: f64, precision: usize) -> String {
    if val.is_infinite() {
        if val.is_sign_positive() {
            "inf".to_string()
        } else {
            "-inf".to_string()
        }
    } else if val.is_nan() {
        "nan".to_string()
    } else {
        format!("{:.prec$}", val, prec = precision)
    }
}

/// Formats scientific/linear ratios cleanly, rendering `+inf` as `inf`, `-inf` as `-inf`,
/// and exact `0.0` as `0.0`.
pub fn format_linear(val: f64) -> String {
    if val.is_infinite() {
        if val.is_sign_positive() {
            "inf".to_string()
        } else {
            "-inf".to_string()
        }
    } else if val.is_nan() {
        "nan".to_string()
    } else if val == 0.0 {
        "0.0".to_string()
    } else {
        format!("{:.6e}", val)
    }
}

/// Prints a structured ASCII table summary of iteration metrics to stdout.
pub fn print_console_summary(metrics: &[IterationMetrics]) {
    let separator = "-".repeat(114);
    println!("{}", separator);
    println!(
        "{:<16} | {:>4} | {:>10} | {:>10} | {:>10} | {:>13} | {:>11} | {:>11} | {:>11}",
        "Chain",
        "Iter",
        "Lag (frm)",
        "SNR (dB)",
        "NSR (dB)",
        "NSR (linear)",
        "LSD Full",
        "LSD Pass",
        "LSD High"
    );
    println!("{}", separator);

    let mut previous_chain: Option<&str> = None;
    for row in metrics {
        if let Some(prev) = previous_chain {
            if prev != row.chain {
                println!("{}", separator);
            }
        }
        previous_chain = Some(&row.chain);

        println!(
            "{:<16} | {:>4} | {:>10} | {:>10} | {:>10} | {:>13} | {:>11} | {:>11} | {:>11}",
            row.chain,
            row.iteration,
            row.lag_frames,
            format_float(row.snr_db, 3),
            format_float(row.nsr_db, 3),
            format_linear(row.nsr_linear),
            format_float(row.lsd_full_db, 4),
            format_float(row.lsd_passband_db, 4),
            format_float(row.lsd_highband_db, 4),
        );
    }
    println!("{}", separator);
}

/// Computes the element-wise average `IterationMetrics` per `(chain, iteration)` across all
/// files in a batch benchmark run.
pub fn compute_average_metrics(records: &[FileBatchRecord]) -> Vec<IterationMetrics> {
    let Some(first) = records.first() else {
        return Vec::new();
    };

    let mut averaged = Vec::with_capacity(first.metrics.len());

    for template in &first.metrics {
        let matching: Vec<&IterationMetrics> = records
            .iter()
            .flat_map(|r| r.metrics.iter())
            .filter(|m| m.chain == template.chain && m.iteration == template.iteration)
            .collect();

        if matching.is_empty() {
            continue;
        }

        let count = matching.len() as f64;
        let avg_lag = (matching.iter().map(|m| m.lag_frames as f64).sum::<f64>() / count).round()
            as isize;

        let finite_snrs: Vec<f64> = matching
            .iter()
            .map(|m| m.snr_db)
            .filter(|v| v.is_finite())
            .collect();
        let avg_snr_db = if finite_snrs.is_empty() {
            f64::INFINITY
        } else {
            finite_snrs.iter().sum::<f64>() / (finite_snrs.len() as f64)
        };

        let finite_nsrs: Vec<f64> = matching
            .iter()
            .map(|m| m.nsr_db)
            .filter(|v| v.is_finite())
            .collect();
        let avg_nsr_db = if finite_nsrs.is_empty() {
            f64::NEG_INFINITY
        } else {
            finite_nsrs.iter().sum::<f64>() / (finite_nsrs.len() as f64)
        };

        let avg_nsr_linear = matching.iter().map(|m| m.nsr_linear).sum::<f64>() / count;
        let avg_lsd_full = matching.iter().map(|m| m.lsd_full_db).sum::<f64>() / count;
        let avg_lsd_pass = matching.iter().map(|m| m.lsd_passband_db).sum::<f64>() / count;
        let avg_lsd_high = matching.iter().map(|m| m.lsd_highband_db).sum::<f64>() / count;

        averaged.push(IterationMetrics {
            chain: template.chain.clone(),
            iteration: template.iteration,
            lag_frames: avg_lag,
            snr_db: avg_snr_db,
            nsr_db: avg_nsr_db,
            nsr_linear: avg_nsr_linear,
            lsd_full_db: avg_lsd_full,
            lsd_passband_db: avg_lsd_pass,
            lsd_highband_db: avg_lsd_high,
        });
    }

    averaged
}

/// Prints the final batch execution summary, per-file runtimes, averaged per-iteration table,
/// and overall cross-iteration averages per codec chain.
pub fn print_batch_summary(
    records: &[FileBatchRecord],
    averaged_metrics: &[IterationMetrics],
    total_batch_secs: f64,
) {
    if records.is_empty() {
        return;
    }

    let file_sep = "=".repeat(88);
    println!("\n{}", file_sep);
    println!("BATCH EXECUTION SUMMARY ({} FILES)", records.len());
    println!("{}", file_sep);
    println!(
        "{:<28} | {:>10} | {:>8} | {:>14} | {:>14}",
        "File", "Rate (Hz)", "Channels", "Audio Len (s)", "Runtime (s)"
    );
    println!("{}", "-".repeat(88));

    let mut sum_file_runtime = 0.0_f64;
    for rec in records {
        let audio_secs = if rec.sample_rate > 0 {
            (rec.num_frames as f64) / f64::from(rec.sample_rate)
        } else {
            0.0
        };
        sum_file_runtime += rec.elapsed_secs;
        println!(
            "{:<28} | {:>10} | {:>8} | {:>14.2} | {:>14.3}",
            rec.file_name, rec.sample_rate, rec.channels, audio_secs, rec.elapsed_secs
        );
    }

    let avg_file_runtime = sum_file_runtime / (records.len() as f64);
    println!("{}", "-".repeat(88));
    println!(
        "Total Batch Runtime: {:.3}s | Average Runtime per File: {:.3}s",
        total_batch_secs, avg_file_runtime
    );
    println!("{}", file_sep);

    println!("\nAVERAGE RESULTS ACROSS ALL {} FILES (BY ITERATION):", records.len());
    print_console_summary(averaged_metrics);

    // Overall average across all iterations and files per chain
    let mut chains: Vec<&str> = Vec::new();
    for m in averaged_metrics {
        if !chains.contains(&m.chain.as_str()) {
            chains.push(&m.chain);
        }
    }

    println!("\nOVERALL CHAIN AVERAGES (ALL FILES & ALL ITERATIONS):");
    let overall_sep = "-".repeat(98);
    println!("{}", overall_sep);
    println!(
        "{:<16} | {:>10} | {:>10} | {:>13} | {:>11} | {:>11} | {:>11}",
        "Chain",
        "Avg SNR",
        "Avg NSR",
        "Avg NSR (lin)",
        "Avg LSD Full",
        "Avg LSD Pass",
        "Avg LSD High"
    );
    println!("{}", overall_sep);

    for chain_name in chains {
        let rows: Vec<&IterationMetrics> = averaged_metrics
            .iter()
            .filter(|m| m.chain == chain_name)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let n = rows.len() as f64;
        let finite_snrs: Vec<f64> = rows
            .iter()
            .map(|r| r.snr_db)
            .filter(|v| v.is_finite())
            .collect();
        let mean_snr = if finite_snrs.is_empty() {
            f64::INFINITY
        } else {
            finite_snrs.iter().sum::<f64>() / (finite_snrs.len() as f64)
        };
        let mean_nsr = -mean_snr;
        let mean_nsr_lin = rows.iter().map(|r| r.nsr_linear).sum::<f64>() / n;
        let mean_full = rows.iter().map(|r| r.lsd_full_db).sum::<f64>() / n;
        let mean_pass = rows.iter().map(|r| r.lsd_passband_db).sum::<f64>() / n;
        let mean_high = rows.iter().map(|r| r.lsd_highband_db).sum::<f64>() / n;

        println!(
            "{:<16} | {:>10} | {:>10} | {:>13} | {:>12} | {:>12} | {:>12}",
            chain_name,
            format_float(mean_snr, 3),
            format_float(mean_nsr, 3),
            format_linear(mean_nsr_lin),
            format_float(mean_full, 4),
            format_float(mean_pass, 4),
            format_float(mean_high, 4),
        );
    }
    println!("{}", overall_sep);
}

/// Writes the iteration-by-iteration degradation curve to a CSV file following the schema:
/// `chain,iteration,lag_frames,snr_db,nsr_db,nsr_linear,lsd_full_db,lsd_passband_db,lsd_highband_db`
pub fn write_csv_report(csv_path: &Path, metrics: &[IterationMetrics]) -> Result<()> {
    if let Some(parent) = csv_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create parent directory '{}'", parent.display())
            })?;
        }
    }

    let file = File::create(csv_path)
        .with_context(|| format!("Failed to create CSV output file '{}'", csv_path.display()))?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "chain,iteration,lag_frames,snr_db,nsr_db,nsr_linear,lsd_full_db,lsd_passband_db,lsd_highband_db"
    )
    .context("Failed to write CSV header")?;

    for row in metrics {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{}",
            row.chain,
            row.iteration,
            row.lag_frames,
            format_float(row.snr_db, 6),
            format_float(row.nsr_db, 6),
            format_linear(row.nsr_linear),
            format_float(row.lsd_full_db, 6),
            format_float(row.lsd_passband_db, 6),
            format_float(row.lsd_highband_db, 6),
        )
        .context("Failed to write CSV data row")?;
    }

    writer
        .flush()
        .with_context(|| format!("Failed to flush CSV file '{}'", csv_path.display()))?;
    Ok(())
}

/// Invokes `scripts/plot_history.py --csv <CSV_OUT> --out <PLOT_OUT>`.
/// Logs a non-fatal warning to `stderr` if `python3` or required plotting libraries are missing.
pub fn trigger_history_plot(csv_path: &Path, plot_path: &Path) {
    let script_path = Path::new("scripts/plot_history.py");
    if !script_path.exists() {
        eprintln!(
            "Warning: Plot script '{}' not found; skipping plot generation.",
            script_path.display()
        );
        return;
    }

    // Prefer project virtual environment python if present, otherwise fall back to python3 on PATH.
    let python_bin = if Path::new(".venv/bin/python3").exists() {
        ".venv/bin/python3"
    } else {
        "python3"
    };

    let output = Command::new(python_bin)
        .arg(script_path)
        .arg("--csv")
        .arg(csv_path)
        .arg("--out")
        .arg(plot_path)
        .output();

    match output {
        Ok(out) if out.status.success() => {
            println!("Saved degradation history plot to '{}'", plot_path.display());
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprintln!(
                "Warning: Could not render degradation plot via '{}' (exit {:?}): {}",
                python_bin,
                out.status.code(),
                stderr.trim()
            );
        }
        Err(err) => {
            eprintln!(
                "Warning: Failed to invoke '{}' for plotting ({}): skipping plot generation.",
                python_bin, err
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_compute_average_metrics() {
        let rec1 = FileBatchRecord {
            file_name: "a.wav".to_string(),
            sample_rate: 48_000,
            channels: 2,
            num_frames: 48_000,
            elapsed_secs: 1.0,
            metrics: vec![IterationMetrics {
                chain: "MP3 <-> Opus".to_string(),
                iteration: 1,
                lag_frames: 10,
                snr_db: 20.0,
                nsr_db: -20.0,
                nsr_linear: 0.01,
                lsd_full_db: 4.0,
                lsd_passband_db: 2.0,
                lsd_highband_db: 10.0,
            }],
        };
        let rec2 = FileBatchRecord {
            file_name: "b.wav".to_string(),
            sample_rate: 48_000,
            channels: 2,
            num_frames: 48_000,
            elapsed_secs: 2.0,
            metrics: vec![IterationMetrics {
                chain: "MP3 <-> Opus".to_string(),
                iteration: 1,
                lag_frames: 20,
                snr_db: 10.0,
                nsr_db: -10.0,
                nsr_linear: 0.03,
                lsd_full_db: 6.0,
                lsd_passband_db: 4.0,
                lsd_highband_db: 14.0,
            }],
        };

        let avg = compute_average_metrics(&[rec1, rec2]);
        assert_eq!(avg.len(), 1);
        assert_eq!(avg[0].lag_frames, 15);
        assert!((avg[0].snr_db - 15.0).abs() < 1e-9);
        assert!((avg[0].lsd_full_db - 5.0).abs() < 1e-9);
        assert!((avg[0].lsd_passband_db - 3.0).abs() < 1e-9);
        assert!((avg[0].lsd_highband_db - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_visualization_smoke() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir for plot test");
        let csv_path = temp_dir.path().join("test_report.csv");
        let png_path = temp_dir.path().join("test_history.png");

        let sample_rows = vec![
            IterationMetrics {
                chain: "WAV <-> FLAC".to_string(),
                iteration: 1,
                lag_frames: 0,
                snr_db: f64::INFINITY,
                nsr_db: f64::NEG_INFINITY,
                nsr_linear: 0.0,
                lsd_full_db: 0.0,
                lsd_passband_db: 0.0,
                lsd_highband_db: 0.0,
            },
            IterationMetrics {
                chain: "WAV <-> FLAC".to_string(),
                iteration: 2,
                lag_frames: 0,
                snr_db: f64::INFINITY,
                nsr_db: f64::NEG_INFINITY,
                nsr_linear: 0.0,
                lsd_full_db: 0.0,
                lsd_passband_db: 0.0,
                lsd_highband_db: 0.0,
            },
            IterationMetrics {
                chain: "MP3 <-> Opus".to_string(),
                iteration: 1,
                lag_frames: 155,
                snr_db: 21.45,
                nsr_db: -21.45,
                nsr_linear: 0.007161,
                lsd_full_db: 6.82,
                lsd_passband_db: 1.94,
                lsd_highband_db: 16.40,
            },
            IterationMetrics {
                chain: "MP3 <-> Opus".to_string(),
                iteration: 2,
                lag_frames: 310,
                snr_db: 18.12,
                nsr_db: -18.12,
                nsr_linear: 0.015417,
                lsd_full_db: 8.15,
                lsd_passband_db: 2.65,
                lsd_highband_db: 19.25,
            },
        ];

        write_csv_report(&csv_path, &sample_rows).expect("Failed to write sample CSV");
        let csv_contents = fs::read_to_string(&csv_path).expect("Failed to read written CSV");
        assert!(csv_contents.contains(",inf,-inf,0.0,"));

        let python_bin = if Path::new(".venv/bin/python3").exists() {
            ".venv/bin/python3"
        } else {
            "python3"
        };

        // Check if required python packages are installed before asserting PNG output.
        let check = Command::new(python_bin)
            .args(["-c", "import matplotlib, numpy, pandas"])
            .output();
        match check {
            Ok(out) if out.status.success() => {}
            _ => {
                eprintln!(
                    "Skipping Python visualization execution in cargo test because matplotlib/numpy/pandas are not installed in '{}'.",
                    python_bin
                );
                return;
            }
        }

        let status = Command::new(python_bin)
            .arg("scripts/plot_history.py")
            .arg("--csv")
            .arg(&csv_path)
            .arg("--out")
            .arg(&png_path)
            .status()
            .expect("Failed to execute scripts/plot_history.py");

        assert!(status.success(), "plot_history.py exited with non-zero status");
        let meta = fs::metadata(&png_path).expect("Expected output PNG file to exist");
        assert!(meta.len() > 0, "Expected non-empty PNG output file");
    }
}
