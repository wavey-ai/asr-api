use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use web_service::{load_default_tls_base64, load_tls_base64_from_paths};

pub const PARAKEET_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "bag-of-beats",
    about = "Upload-response backed Parakeet TDT transcription service"
)]
pub struct AppConfig {
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    pub rust_log: String,

    #[arg(long, env = "PORT", default_value_t = 8443)]
    pub port: u16,

    #[arg(long, env = "ENABLE_H3", default_value_t = false)]
    pub enable_h3: bool,

    #[arg(long, env = "TLS_CERT_PATH")]
    pub tls_cert_path: Option<PathBuf>,

    #[arg(long, env = "TLS_KEY_PATH")]
    pub tls_key_path: Option<PathBuf>,

    #[arg(long, env = "PARAKEET_MODEL_DIR")]
    pub model_dir: PathBuf,

    #[arg(long, env = "MODEL_INSTANCES", default_value_t = 1)]
    pub model_instances: usize,

    #[arg(long, env = "UPLOAD_NUM_STREAMS", default_value_t = 32)]
    pub num_streams: usize,

    #[arg(long, env = "UPLOAD_SLOT_SIZE_KB", default_value_t = 64)]
    pub slot_size_kb: usize,

    #[arg(long, env = "UPLOAD_SLOTS_PER_STREAM", default_value_t = 32_768)]
    pub slots_per_stream: usize,

    #[arg(long, env = "RESPONSE_IDLE_TIMEOUT_MS", default_value_t = 300_000)]
    pub response_timeout_ms: u64,

    #[arg(long, env = "CHUNK_SECONDS", default_value_t = 30.0)]
    pub chunk_seconds: f32,

    #[arg(long, env = "OVERLAP_SECONDS", default_value_t = 2.0)]
    pub overlap_seconds: f32,

    #[arg(long, env = "FINAL_MIN_SECONDS", default_value_t = 0.5)]
    pub final_min_seconds: f32,
}

impl AppConfig {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.model_dir.is_dir(),
            "PARAKEET_MODEL_DIR must point to a directory"
        );
        anyhow::ensure!(self.model_instances > 0, "MODEL_INSTANCES must be > 0");
        anyhow::ensure!(self.num_streams > 0, "UPLOAD_NUM_STREAMS must be > 0");
        anyhow::ensure!(self.slot_size_kb > 0, "UPLOAD_SLOT_SIZE_KB must be > 0");
        anyhow::ensure!(
            self.slots_per_stream > 2,
            "UPLOAD_SLOTS_PER_STREAM must be > 2"
        );
        anyhow::ensure!(self.chunk_seconds > 0.0, "CHUNK_SECONDS must be > 0");
        anyhow::ensure!(self.overlap_seconds >= 0.0, "OVERLAP_SECONDS must be >= 0");
        anyhow::ensure!(
            self.chunk_seconds > self.overlap_seconds,
            "CHUNK_SECONDS must be larger than OVERLAP_SECONDS"
        );
        anyhow::ensure!(
            self.final_min_seconds >= 0.0,
            "FINAL_MIN_SECONDS must be >= 0"
        );
        Ok(())
    }

    pub fn tls_base64(&self) -> Result<(String, String)> {
        match (&self.tls_cert_path, &self.tls_key_path) {
            (Some(cert), Some(key)) => load_tls_base64_from_paths(cert, key).with_context(|| {
                format!(
                    "failed to load TLS PEMs from {} and {}",
                    cert.display(),
                    key.display()
                )
            }),
            (None, None) => load_default_tls_base64()
                .context("failed to load default local Wavey TLS certificate"),
            _ => anyhow::bail!("set both TLS_CERT_PATH and TLS_KEY_PATH, or neither"),
        }
    }

    pub fn chunk_samples(&self) -> usize {
        seconds_to_samples(self.chunk_seconds)
    }

    pub fn overlap_samples(&self) -> usize {
        seconds_to_samples(self.overlap_seconds)
    }

    pub fn min_final_samples(&self) -> usize {
        seconds_to_samples(self.final_min_seconds)
    }
}

fn seconds_to_samples(seconds: f32) -> usize {
    (seconds.max(0.0) * PARAKEET_SAMPLE_RATE as f32).round() as usize
}
