use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::env;
use std::path::{Path, PathBuf};
use upload_response::UploadResponseConfig;
use web_service::{load_default_tls_base64, load_tls_base64_from_paths};

pub const ASR_SAMPLE_RATE: u32 = 16_000;
pub const DEFAULT_COHERE_ONNX_MODEL_NAME: &str = "wavey-cohere-transcribe-onnx";
pub const DEFAULT_COHERE_MLX_MODEL_NAME: &str = "wavey-cohere-transcribe-mlx";
pub const DEFAULT_COHERE_MODEL_NAME: &str = DEFAULT_COHERE_ONNX_MODEL_NAME;
pub const DEFAULT_PARAKEET_ONNX_MODEL_NAME: &str = "wavey-parakeet-tdt-onnx";
pub const DEFAULT_MODEL_NAME: &str = DEFAULT_COHERE_MODEL_NAME;
pub const DEFAULT_LANGUAGE: &str = "en";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LogFormat {
    Json,
    Pretty,
    Compact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum AppRole {
    Ingress,
    Decoder,
    Worker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum AsrModelProvider {
    Auto,
    Cohere,
    Parakeet,
}

impl AsrModelProvider {
    pub fn default_model_name(self) -> &'static str {
        match self {
            Self::Auto | Self::Cohere => default_cohere_model_name(),
            Self::Parakeet => DEFAULT_PARAKEET_ONNX_MODEL_NAME,
        }
    }

    pub fn default_model_arch(self) -> &'static str {
        match self {
            Self::Auto | Self::Cohere => "cohere-transcribe-seq2seq",
            Self::Parakeet => "parakeet-tdt",
        }
    }
}

pub fn default_cohere_model_name() -> &'static str {
    cohere_model_name_for_backend(env::var("ASR_COHERE_BACKEND").ok().as_deref())
}

impl AppRole {
    pub fn uses_asr_backend(self) -> bool {
        matches!(self, Self::Worker)
    }

    pub fn uses_audio_decoder(self) -> bool {
        matches!(self, Self::Decoder)
    }

    pub fn serves_listen(self) -> bool {
        matches!(self, Self::Ingress)
    }

    pub fn exposes_upload_cache(self) -> bool {
        matches!(self, Self::Ingress)
    }
}

#[derive(Debug, Clone, Parser)]
#[command(
    name = "asr-api",
    about = "Deepgram-compatible ASR service over Wavey's web-service stack"
)]
pub struct AppConfig {
    #[arg(long, env = "ASR_API_ROLE", value_enum, default_value_t = AppRole::Ingress)]
    pub role: AppRole,

    #[arg(long, env = "RUST_LOG", default_value = "info")]
    pub rust_log: String,

    #[arg(long, env = "ASR_LOG_FORMAT", value_enum, default_value_t = LogFormat::Json)]
    pub log_format: LogFormat,

    #[arg(long, env = "PORT", default_value_t = 8443)]
    pub port: u16,

    #[arg(long, env = "ENABLE_H3", default_value_t = false)]
    pub enable_h3: bool,

    #[arg(long, env = "TLS_CERT_PATH")]
    pub tls_cert_path: Option<PathBuf>,

    #[arg(long, env = "TLS_KEY_PATH")]
    pub tls_key_path: Option<PathBuf>,

    #[arg(long = "model-dir", env = "ASR_MODEL_DIR")]
    pub model_dir: Option<PathBuf>,

    #[arg(
        long = "model-provider",
        env = "ASR_MODEL_PROVIDER",
        value_enum,
        default_value_t = AsrModelProvider::Auto
    )]
    pub model_provider: AsrModelProvider,

    #[arg(
        long,
        env = "ASR_DEVICE_IDS",
        value_delimiter = ',',
        default_value = "0"
    )]
    pub device_ids: Vec<usize>,

    #[arg(long, env = "ASR_ONNX_SESSIONS", default_value_t = 1)]
    pub onnx_sessions: usize,

    #[arg(long, env = "ASR_COHERE_MAX_NEW_TOKENS", default_value_t = 384)]
    pub cohere_max_new_tokens: usize,

    #[arg(long, env = "CHUNK_SECONDS", default_value_t = 30.0)]
    pub chunk_seconds: f32,

    #[arg(long, env = "OVERLAP_SECONDS", default_value_t = 2.0)]
    pub overlap_seconds: f32,

    #[arg(long, env = "FINAL_MIN_SECONDS", default_value_t = 0.5)]
    pub final_min_seconds: f32,

    #[arg(long, env = "UTT_SPLIT_SECONDS", default_value_t = 0.8)]
    pub utt_split_seconds: f64,

    #[arg(long, env = "UPLOAD_RESPONSE_NUM_STREAMS", default_value_t = 16)]
    pub upload_response_num_streams: usize,

    #[arg(long, env = "UPLOAD_RESPONSE_SLOT_SIZE_KB", default_value_t = 32)]
    pub upload_response_slot_size_kb: usize,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_SLOTS_PER_STREAM",
        default_value_t = 1_024
    )]
    pub upload_response_slots_per_stream: usize,

    #[arg(long, env = "UPLOAD_RESPONSE_TIMEOUT_MS", default_value_t = 30_000)]
    pub upload_response_timeout_ms: u64,

    #[arg(long, env = "UPLOAD_RESPONSE_WATCH_POLL_MS", default_value_t = 1)]
    pub upload_response_watch_poll_ms: u64,

    #[arg(long, env = "UPLOAD_RESPONSE_WORKER_POLL_MS", default_value_t = 2)]
    pub upload_response_worker_poll_ms: u64,

    #[arg(long, env = "UPLOAD_RESPONSE_MAX_INFLIGHT", default_value_t = 2)]
    pub upload_response_max_inflight: usize,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_WORKER_ID",
        default_value = "asr-api-worker"
    )]
    pub upload_response_worker_id: String,

    #[arg(long, env = "UPLOAD_RESPONSE_INGRESS_URLS", value_delimiter = ',')]
    pub upload_response_ingress_urls: Vec<String>,

    #[arg(long, env = "UPLOAD_RESPONSE_DISCOVERY_DNS")]
    pub upload_response_discovery_dns: Option<String>,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_DISCOVERY_INTERVAL_MS",
        default_value_t = 2_000
    )]
    pub upload_response_discovery_interval_ms: u64,

    #[arg(long, env = "UPLOAD_RESPONSE_INSECURE_TLS", default_value_t = false)]
    pub upload_response_insecure_tls: bool,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_WORKER_HEARTBEAT_INTERVAL_MS",
        default_value_t = 1_000
    )]
    pub upload_response_worker_heartbeat_interval_ms: u64,

    #[arg(long, env = "UPLOAD_RESPONSE_WORKER_TTL_MS", default_value_t = 5_000)]
    pub upload_response_worker_ttl_ms: u64,
}

impl AppConfig {
    pub fn validate(&self) -> Result<()> {
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
        anyhow::ensure!(
            self.utt_split_seconds >= 0.0,
            "UTT_SPLIT_SECONDS must be >= 0"
        );
        anyhow::ensure!(
            self.upload_response_num_streams > 0,
            "UPLOAD_RESPONSE_NUM_STREAMS must be > 0"
        );
        anyhow::ensure!(
            self.upload_response_slot_size_kb > 0,
            "UPLOAD_RESPONSE_SLOT_SIZE_KB must be > 0"
        );
        anyhow::ensure!(
            self.upload_response_slots_per_stream > 0,
            "UPLOAD_RESPONSE_SLOTS_PER_STREAM must be > 0"
        );
        anyhow::ensure!(
            self.upload_response_timeout_ms > 0,
            "UPLOAD_RESPONSE_TIMEOUT_MS must be > 0"
        );
        anyhow::ensure!(
            self.upload_response_max_inflight > 0,
            "UPLOAD_RESPONSE_MAX_INFLIGHT must be > 0"
        );
        anyhow::ensure!(
            !self.upload_response_worker_id.trim().is_empty(),
            "UPLOAD_RESPONSE_WORKER_ID must not be empty"
        );
        anyhow::ensure!(
            self.upload_response_discovery_interval_ms > 0,
            "UPLOAD_RESPONSE_DISCOVERY_INTERVAL_MS must be > 0"
        );
        anyhow::ensure!(
            self.upload_response_worker_heartbeat_interval_ms > 0,
            "UPLOAD_RESPONSE_WORKER_HEARTBEAT_INTERVAL_MS must be > 0"
        );
        anyhow::ensure!(
            self.upload_response_worker_ttl_ms > 0,
            "UPLOAD_RESPONSE_WORKER_TTL_MS must be > 0"
        );

        if self.role.uses_asr_backend() {
            let model_dir = self.model_dir()?;
            let provider = self.resolved_model_provider()?;
            anyhow::ensure!(
                self.onnx_sessions > 0,
                "ASR_ONNX_SESSIONS must be greater than 0"
            );
            anyhow::ensure!(
                self.cohere_max_new_tokens > 0,
                "ASR_COHERE_MAX_NEW_TOKENS must be greater than 0"
            );

            match provider {
                AsrModelProvider::Cohere => {
                    if cohere_runtime_is_mlx() {
                        ensure_all_exists(
                            model_dir,
                            &["config.json", "model.safetensors", "vocab.json"],
                        )?;
                    } else {
                        let force_cpu = env_var_truthy("ASR_COHERE_FORCE_CPU");
                        let coreml = env_var_truthy("ASR_COHERE_COREML")
                            || env::var("ASR_COHERE_EXECUTION_PROVIDER")
                                .ok()
                                .map(|value| {
                                    matches!(
                                        value.trim().to_ascii_lowercase().as_str(),
                                        "coreml" | "metal" | "apple"
                                    )
                                })
                                .unwrap_or(false);
                        anyhow::ensure!(
                            force_cpu || coreml || !self.device_ids.is_empty(),
                            "Cohere ONNX backend requires at least one GPU device id; set ASR_DEVICE_IDS, ASR_COHERE_COREML=true for Apple GPU/CoreML, or ASR_COHERE_FORCE_CPU=true for explicit CPU compare mode"
                        );
                        ensure_all_exists(
                            model_dir,
                            &[
                                "encoder.onnx",
                                "encoder.onnx.data",
                                "decoder_prefill.onnx",
                                "decoder_prefill.onnx.data",
                                "decoder_cached_step.onnx",
                                "decoder_cached_step.onnx.data",
                                "tokenizer.json",
                                "tokenizer.model",
                                "config.json",
                                "generation_config.json",
                                "preprocessor_config.json",
                            ],
                        )?;
                    }
                }
                AsrModelProvider::Parakeet => {
                    ensure_all_exists(
                        model_dir,
                        &[
                            "encoder.onnx",
                            "decoder.onnx",
                            "joint.enc.onnx",
                            "joint.pred.onnx",
                            "joint.joint_net.onnx",
                            "tokens.txt",
                        ],
                    )?;
                }
                AsrModelProvider::Auto => {
                    unreachable!("model provider should resolve before validation")
                }
            }
        }

        if matches!(self.role, AppRole::Decoder | AppRole::Worker) {
            anyhow::ensure!(
                !self.upload_response_ingress_urls.is_empty()
                    || self
                        .upload_response_discovery_dns
                        .as_ref()
                        .map(|value| !value.trim().is_empty())
                        .unwrap_or(false),
                "worker role requires UPLOAD_RESPONSE_INGRESS_URLS or UPLOAD_RESPONSE_DISCOVERY_DNS"
            );
        }

        Ok(())
    }

    pub fn configured_model_provider(&self) -> AsrModelProvider {
        match self.model_provider {
            AsrModelProvider::Auto => AsrModelProvider::Cohere,
            provider => provider,
        }
    }

    pub fn default_model_provider(&self) -> AsrModelProvider {
        self.resolved_model_provider()
            .unwrap_or_else(|_| self.configured_model_provider())
    }

    pub fn default_model_name(&self) -> &'static str {
        self.default_model_provider().default_model_name()
    }

    pub fn default_model_arch(&self) -> &'static str {
        self.default_model_provider().default_model_arch()
    }

    pub fn resolved_model_provider(&self) -> Result<AsrModelProvider> {
        match self.model_provider {
            AsrModelProvider::Auto => Ok(AsrModelProvider::Cohere),
            provider => Ok(provider),
        }
    }

    pub fn model_dir(&self) -> Result<&Path> {
        let model_dir = self
            .model_dir
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("ASR_MODEL_DIR is required for this asr-api role"))?;
        anyhow::ensure!(
            model_dir.is_dir(),
            "ASR_MODEL_DIR must point to a directory"
        );
        Ok(model_dir)
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

    pub fn upload_response_config(&self) -> UploadResponseConfig {
        UploadResponseConfig {
            num_streams: self.upload_response_num_streams,
            slot_size_kb: self.upload_response_slot_size_kb,
            slots_per_stream: self.upload_response_slots_per_stream,
            response_timeout_ms: self.upload_response_timeout_ms,
        }
    }
}

fn ensure_all_exists(model_dir: &Path, required: &[&str]) -> Result<()> {
    let missing = required
        .iter()
        .copied()
        .filter(|name| !model_dir.join(name).is_file())
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }

    anyhow::bail!(
        "missing required model files in {}: {}",
        model_dir.display(),
        missing.join(", ")
    )
}

fn seconds_to_samples(seconds: f32) -> usize {
    (seconds.max(0.0) * ASR_SAMPLE_RATE as f32).round() as usize
}

fn cohere_runtime_is_mlx() -> bool {
    env::var("ASR_COHERE_BACKEND")
        .ok()
        .map(|value| value.trim().eq_ignore_ascii_case("mlx"))
        .unwrap_or(false)
}

fn cohere_model_name_for_backend(backend: Option<&str>) -> &'static str {
    if backend
        .map(|value| value.trim().eq_ignore_ascii_case("mlx"))
        .unwrap_or(false)
    {
        DEFAULT_COHERE_MLX_MODEL_NAME
    } else {
        DEFAULT_COHERE_ONNX_MODEL_NAME
    }
}

fn env_var_truthy(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "True"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohere_default_model_name_matches_runtime_label() {
        assert_eq!(
            cohere_model_name_for_backend(None),
            DEFAULT_COHERE_ONNX_MODEL_NAME
        );
        assert_eq!(
            cohere_model_name_for_backend(Some("onnx")),
            DEFAULT_COHERE_ONNX_MODEL_NAME
        );
        assert_eq!(
            cohere_model_name_for_backend(Some("mlx")),
            DEFAULT_COHERE_MLX_MODEL_NAME
        );
    }
}
