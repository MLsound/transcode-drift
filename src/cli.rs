use std::path::PathBuf;

use clap::{Parser, ValueEnum};

/// Codec chain selection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ChainSelection {
    /// Run both lossless (WAV <-> FLAC) and lossy (MP3 <-> Opus) chains.
    Both,
    /// Run only the lossless baseline chain (WAV <-> FLAC).
    #[value(alias = "wav-flac", alias = "flac")]
    Lossless,
    /// Run only the lossy web-transfer chain (MP3 <-> Opus).
    #[value(alias = "mp3-opus", alias = "opus", alias = "mp3")]
    Lossy,
}

/// Measure cumulative audio quality degradation across iterative codec transcoding cycles.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "codec-degradation",
    version,
    about = "Benchmark cumulative time- and frequency-domain audio degradation across iterative round-trip transcodes"
)]
pub struct CliArgs {
    /// Path to input reference WAV file, or a folder of WAV files for batch testing.
    /// If omitted, generates a 3-second 48 kHz stereo test signal.
    #[arg(long)]
    pub input: Option<PathBuf>,

    /// Codec chain(s) to execute: both, lossless (wav-flac), or lossy (mp3-opus).
    #[arg(long, value_enum, default_value_t = ChainSelection::Both)]
    pub chain: ChainSelection,

    /// Number of round-trip conversion cycles.
    #[arg(long, default_value_t = 10)]
    pub iterations: usize,

    /// Target MP3 bitrate in kbps.
    #[arg(long, default_value_t = 192)]
    pub mp3_kbps: u32,

    /// Target Opus bitrate in kbps.
    #[arg(long, default_value_t = 128)]
    pub opus_kbps: u32,

    /// Maximum frame search window for cross-correlation alignment.
    #[arg(long, default_value_t = 2048)]
    pub max_lag_frames: isize,

    /// Output path for per-iteration CSV results.
    #[arg(long, default_value = "degradation_report.csv")]
    pub csv_out: PathBuf,

    /// Optional output path for rendering the multi-panel PNG history plot.
    #[arg(long)]
    pub plot_out: Option<PathBuf>,

    /// Persist intermediate audio files to this directory instead of auto-deleting.
    #[arg(long)]
    pub keep_temp: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_chain_selection_parsing() {
        let default_args =
            CliArgs::try_parse_from(["codec-degradation"]).expect("default CLI args should parse");
        assert_eq!(default_args.chain, ChainSelection::Both);

        let lossless_args =
            CliArgs::try_parse_from(["codec-degradation", "--chain", "lossless"])
                .expect("--chain lossless should parse");
        assert_eq!(lossless_args.chain, ChainSelection::Lossless);

        let wav_flac_alias =
            CliArgs::try_parse_from(["codec-degradation", "--chain", "wav-flac"])
                .expect("--chain wav-flac alias should parse");
        assert_eq!(wav_flac_alias.chain, ChainSelection::Lossless);

        let lossy_args = CliArgs::try_parse_from(["codec-degradation", "--chain", "lossy"])
            .expect("--chain lossy should parse");
        assert_eq!(lossy_args.chain, ChainSelection::Lossy);

        let mp3_opus_alias =
            CliArgs::try_parse_from(["codec-degradation", "--chain", "mp3-opus"])
                .expect("--chain mp3-opus alias should parse");
        assert_eq!(mp3_opus_alias.chain, ChainSelection::Lossy);
    }
}

