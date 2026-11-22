# transcode-drift

A Rust CLI benchmark and Python visualization suite that measures cumulative audio quality loss across $N$ iterative round-trip transcodes (`A -> B -> A`) for two codec chains:

1. **Lossless baseline:** `WAV <-> FLAC`
2. **Lossy web-transfer chain:** `MP3 <-> Opus`

After every iteration $i \in [1..N]$, the benchmark compensates for encoder priming delay via overlap-normalized cross-correlation, logs per-iteration execution runtime, computes time-domain SNR/NSR and three-band STFT Log-Spectral Distance (LSD), prints a console summary table, exports a structured CSV report, and optionally renders a 3-panel matplotlib PNG figure.

## Quick Path

1. Ensure `ffmpeg` (compiled with `libmp3lame` and `libopus`) is installed and available on `PATH`.
2. Build and run the benchmark (uses a built-in 3-second 48 kHz stereo synthetic signal when `--input` is omitted):
   ```bash
   cargo run --release -- --iterations 10 --csv-out degradation_report.csv --plot-out degradation_history.png
   ```
3. Run the test suite and linter:
   ```bash
   cargo test
   cargo clippy -- -D warnings
   ```

## Usage Examples

### Run Only One Chain (`lossless` or `lossy`)

```bash
# Run only the MP3 <-> Opus lossy chain
cargo run --release -- --chain lossy --input media48/bartok.wav --iterations 10 --plot-out degradation_history.png

# Run only the WAV <-> FLAC lossless baseline chain
cargo run --release -- --chain lossless --input media48/bartok.wav --iterations 10
```

### Batch Testing on a Folder of WAV Files

Pass a directory to `--input` to count all `.wav` files, prompt for confirmation, execute the selected chain(s) across every file while reporting per-iteration and per-file runtimes, and print an end-of-run summary and batch average table:

```bash
cargo run --release -- \
  --input media48 \
  --chain both \
  --iterations 10 \
  --csv-out degradation_report.csv \
  --plot-out degradation_history.png
```

### Run with a Single Custom Reference WAV

```bash
cargo run --release -- \
  --input media48/bartok.wav \
  --iterations 15 \
  --mp3-kbps 192 \
  --opus-kbps 128 \
  --max-lag-frames 2048 \
  --csv-out degradation_report.csv \
  --plot-out degradation_history.png
```

### Render or Re-render the History Chart Directly

```bash
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
.venv/bin/python3 scripts/plot_history.py --csv degradation_report.csv --out degradation_history.png
```

## Architecture & Technical Decisions

| Area | Decision |
|------|----------|
| Priming delay alignment | Searches integer frame lags $\tau \in [-\text{max\_lag\_frames}, +\text{max\_lag\_frames}]$ stepped by `channels` and maximizes mean cross-correlation normalized by overlap length $M_\tau$. |
| Time-domain metrics | Evaluates aligned slices for signal and error power; returns exact `inf` SNR and `-inf` NSR when noise power is zero. |
| Spectral distortion (LSD) | De-interleaves channels, runs a 2048-point Hann-windowed STFT (`hop_size = 512`), gates silent frames at $-60\text{ dB}$ relative to peak reference frame power, and evaluates Full (`20 Hz - 20 kHz`), Passband (`20 Hz - 16 kHz`), and Highband (`16 kHz - 20 kHz`) LSD. |
| Batch folder mode | Discovers `.wav` files, prompts for confirmation before starting, tracks per-iteration and per-file wall-clock runtime, and computes per-iteration and overall chain averages across all files. |

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--input <PATH>` | Built-in 3s 48 kHz stereo signal | Path to a reference PCM WAV file or a folder of `.wav` files for batch mode. |
| `--chain <CHAIN>` | `both` | Codec chain(s) to run: `both`, `lossless` (alias `wav-flac`), or `lossy` (alias `mp3-opus`). |
| `--iterations <USIZE>` | `10` | Number of round-trip conversion cycles per chain. |
| `--mp3-kbps <U32>` | `192` | Target MP3 bitrate (`libmp3lame`) in kbps. |
| `--opus-kbps <U32>` | `128` | Target Opus bitrate (`libopus`) in kbps. |
| `--max-lag-frames <ISIZE>` | `2048` | Maximum frame lag search window for cross-correlation alignment. |
| `--csv-out <PATH>` | `degradation_report.csv` | Output path for per-iteration (or batch-averaged) CSV metrics. |
| `--plot-out <PATH>` | None | Optional PNG output path; triggers `scripts/plot_history.py`. |
| `--keep-temp <PATH>` | Auto-deleted `TempDir` | Directory to persist intermediate transcoded files. |

## Test Suite Coverage

Run `cargo test` to execute all unit and integration tests:

| Test | Module | Verified Behavior |
|------|--------|-------------------|
| `test_cli_chain_selection_parsing` | `src/cli.rs` | Parses `--chain` variants (`both`, `lossless`, `wav-flac`, `lossy`, `mp3-opus`). |
| `test_discover_wav_files_and_confirmation_parsing` | `src/main.rs` | Counts, filters, and sorts `.wav` files in a batch folder and validates confirmation responses. |
| `test_bit_exact_identity` | `src/metrics.rs` | Confirms zero lag, `inf` SNR, `0.0` linear NSR, and `< 1e-9` LSD on identical signals. |
| `test_priming_delay_recovery` | `src/metrics.rs` | Recovers an artificial `+43` stereo frame shift (`86` samples) with `SNR > 100 dB`. |
| `test_band_limited_spectral_isolation` | `src/metrics.rs` | Isolates `> 16.5 kHz` attenuation to `lsd_highband_db > 15.0` while `lsd_passband_db < 0.5`. |
| `test_end_to_end_smoke_2_iterations` | `src/transcode.rs` | Runs 2 full iterations of both `WAV <-> FLAC` and `MP3 <-> Opus` via `ffmpeg`. |
| `test_single_chain_selection` | `src/transcode.rs` | Verifies isolated execution of `ChainSelection::Lossless` and `ChainSelection::Lossy`. |
| `test_compute_average_metrics` | `src/report.rs` | Verifies multi-file batch averaging of lag, SNR, NSR, and three-band LSD across files. |
| `test_visualization_smoke` | `src/report.rs` | Verifies CSV serialization (`inf`/`-inf`) and PNG chart output via `scripts/plot_history.py`. |

