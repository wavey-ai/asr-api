use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::time::{interval, sleep, Duration};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ModelProvider {
    Auto,
    Nemo,
    Cohere,
}

impl ModelProvider {
    fn as_env(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Nemo => "nemo",
            Self::Cohere => "cohere",
        }
    }
}

#[derive(Debug, Clone, Parser)]
#[command(
    name = "local-orchestrator",
    about = "Launch a local ingress/decoder/worker asr-api stack"
)]
struct Config {
    #[arg(long)]
    asr_api_bin: Option<PathBuf>,

    #[arg(long, env = "ASR_MODEL_DIR")]
    model_dir: PathBuf,

    #[arg(long, env = "ASR_VOCAB_PATH")]
    vocab_path: Option<PathBuf>,

    #[arg(
        long,
        env = "ASR_MODEL_PROVIDER",
        value_enum,
        default_value_t = ModelProvider::Auto
    )]
    model_provider: ModelProvider,

    #[arg(long, env = "ASR_DEVICE_IDS", default_value = "0")]
    device_ids: String,

    #[arg(long, env = "ASR_TORCH_SESSIONS", default_value_t = 1)]
    torch_sessions: usize,

    #[arg(long, env = "ASR_ONNX_SESSIONS", default_value_t = 1)]
    onnx_sessions: usize,

    #[arg(long, env = "ASR_COHERE_MAX_NEW_TOKENS", default_value_t = 384)]
    cohere_max_new_tokens: usize,

    #[arg(long, env = "RUST_LOG", default_value = "info")]
    rust_log: String,

    #[arg(long, env = "ASR_LOG_FORMAT", default_value = "compact")]
    log_format: String,

    #[arg(long, env = "ENABLE_H3", default_value_t = false)]
    enable_h3: bool,

    #[arg(long, env = "TLS_CERT_PATH")]
    tls_cert_path: Option<PathBuf>,

    #[arg(long, env = "TLS_KEY_PATH")]
    tls_key_path: Option<PathBuf>,

    #[arg(long, default_value_t = 8443)]
    ingress_port: u16,

    #[arg(long, default_value_t = 9443)]
    decoder_port: u16,

    #[arg(long, default_value_t = 10443)]
    worker_port: u16,

    #[arg(long, env = "ASR_WORKER_COUNT", default_value_t = 1)]
    worker_count: usize,

    #[arg(long, env = "CHUNK_SECONDS", default_value_t = 30.0)]
    chunk_seconds: f32,

    #[arg(long, env = "OVERLAP_SECONDS", default_value_t = 2.0)]
    overlap_seconds: f32,

    #[arg(long, env = "FINAL_MIN_SECONDS", default_value_t = 0.5)]
    final_min_seconds: f32,

    #[arg(long, env = "UTT_SPLIT_SECONDS", default_value_t = 0.8)]
    utt_split_seconds: f64,

    #[arg(long, env = "UPLOAD_RESPONSE_NUM_STREAMS", default_value_t = 16)]
    upload_response_num_streams: usize,

    #[arg(long, env = "UPLOAD_RESPONSE_SLOT_SIZE_KB", default_value_t = 32)]
    upload_response_slot_size_kb: usize,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_SLOTS_PER_STREAM",
        default_value_t = 1_024
    )]
    upload_response_slots_per_stream: usize,

    #[arg(long, env = "UPLOAD_RESPONSE_TIMEOUT_MS", default_value_t = 30_000)]
    upload_response_timeout_ms: u64,

    #[arg(long, env = "UPLOAD_RESPONSE_WATCH_POLL_MS", default_value_t = 1)]
    upload_response_watch_poll_ms: u64,

    #[arg(long, env = "UPLOAD_RESPONSE_WORKER_POLL_MS", default_value_t = 2)]
    upload_response_worker_poll_ms: u64,

    #[arg(long, env = "UPLOAD_RESPONSE_MAX_INFLIGHT", default_value_t = 2)]
    upload_response_max_inflight: usize,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_WORKER_ID_PREFIX",
        default_value = "asr-api-worker-local"
    )]
    upload_response_worker_id_prefix: String,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_DISCOVERY_INTERVAL_MS",
        default_value_t = 2_000
    )]
    upload_response_discovery_interval_ms: u64,

    #[arg(long, env = "UPLOAD_RESPONSE_INSECURE_TLS", default_value_t = true)]
    upload_response_insecure_tls: bool,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_WORKER_HEARTBEAT_INTERVAL_MS",
        default_value_t = 1_000
    )]
    upload_response_worker_heartbeat_interval_ms: u64,

    #[arg(long, env = "UPLOAD_RESPONSE_WORKER_TTL_MS", default_value_t = 5_000)]
    upload_response_worker_ttl_ms: u64,
}

struct Service {
    name: String,
    child: Child,
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::parse();
    validate_config(&config)?;
    let asr_api_bin = ensure_asr_api_binary(config.asr_api_bin.clone()).await?;
    let ingress_url = format!("https://127.0.0.1:{}", config.ingress_port);
    let worker_urls = worker_urls(&config)?;

    let mut services = Vec::with_capacity(2 + config.worker_count);
    services.push(
        spawn_service(
            "ingress",
            &asr_api_bin,
            &config,
            vec![
                ("ASR_API_ROLE", "ingress".to_string()),
                ("PORT", config.ingress_port.to_string()),
            ],
        )
        .await?,
    );
    services.push(
        spawn_service(
            "decoder",
            &asr_api_bin,
            &config,
            vec![
                ("ASR_API_ROLE", "decoder".to_string()),
                ("PORT", config.decoder_port.to_string()),
                (
                    "UPLOAD_RESPONSE_WORKER_ID",
                    "asr-api-decoder-local".to_string(),
                ),
                ("UPLOAD_RESPONSE_INGRESS_URLS", ingress_url.clone()),
            ],
        )
        .await?,
    );

    for worker_index in 0..config.worker_count {
        let port = worker_port(&config, worker_index)?;
        services.push(
            spawn_service(
                worker_service_name(&config, worker_index),
                &asr_api_bin,
                &config,
                vec![
                    ("ASR_API_ROLE", "worker".to_string()),
                    ("PORT", port.to_string()),
                    ("ASR_MODEL_DIR", config.model_dir.display().to_string()),
                    (
                        "ASR_MODEL_PROVIDER",
                        config.model_provider.as_env().to_string(),
                    ),
                    ("ASR_TORCH_SESSIONS", config.torch_sessions.to_string()),
                    ("ASR_ONNX_SESSIONS", config.onnx_sessions.to_string()),
                    (
                        "ASR_COHERE_MAX_NEW_TOKENS",
                        config.cohere_max_new_tokens.to_string(),
                    ),
                    ("ASR_DEVICE_IDS", config.device_ids.clone()),
                    (
                        "UPLOAD_RESPONSE_WORKER_ID",
                        worker_id(&config, worker_index),
                    ),
                    ("UPLOAD_RESPONSE_INGRESS_URLS", ingress_url.clone()),
                ],
            )
            .await?,
        );
    }

    eprintln!(
        "local stack ready: ingress=https://127.0.0.1:{} decoder=https://127.0.0.1:{} workers={}",
        config.ingress_port,
        config.decoder_port,
        worker_urls.join(",")
    );

    let mut tick = interval(Duration::from_millis(250));
    let mut exit_reason = None;

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                exit_reason = Some("received Ctrl-C".to_string());
                break;
            }
            _ = tick.tick() => {
                for service in &mut services {
                    if let Some(status) = service.child.try_wait().with_context(|| {
                        format!("failed to poll {} process", service.name)
                    })? {
                        exit_reason = Some(format!("{} exited with {}", service.name, status));
                        break;
                    }
                }
                if exit_reason.is_some() {
                    break;
                }
            }
        }
    }

    shutdown_services(&mut services).await;

    if let Some(reason) = exit_reason {
        eprintln!("{reason}");
    }

    Ok(())
}

async fn ensure_asr_api_binary(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        anyhow::ensure!(
            path.is_file(),
            "ASR API binary does not exist: {}",
            path.display()
        );
        return Ok(path);
    }

    let current_exe = std::env::current_exe().context("failed to resolve current executable")?;
    let profile_dir = current_exe
        .parent()
        .context("current executable has no parent directory")?;
    let asr_api_bin = profile_dir.join(exe_name("asr-api"));
    if asr_api_bin.is_file() {
        return Ok(asr_api_bin);
    }

    let profile = profile_dir
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("debug");
    let mut command = Command::new("cargo");
    command.arg("build").arg("--bin").arg("asr-api");
    if profile == "release" {
        command.arg("--release");
    }
    command.current_dir(env!("CARGO_MANIFEST_DIR"));
    let status = command
        .status()
        .await
        .context("failed to build sibling asr-api binary")?;
    anyhow::ensure!(status.success(), "cargo build --bin asr-api failed");
    anyhow::ensure!(
        asr_api_bin.is_file(),
        "expected asr-api binary at {} after build",
        asr_api_bin.display()
    );
    Ok(asr_api_bin)
}

async fn spawn_service(
    name: impl Into<String>,
    asr_api_bin: &Path,
    config: &Config,
    role_env: Vec<(&'static str, String)>,
) -> Result<Service> {
    let name = name.into();
    let mut command = Command::new(asr_api_bin);
    command.stdin(Stdio::null());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    command.env_remove("ASR_DEVICE_IDS");
    command.env("RUST_LOG", &config.rust_log);
    command.env("ASR_LOG_FORMAT", &config.log_format);
    command.env("ENABLE_H3", bool_env(config.enable_h3));
    command.env("CHUNK_SECONDS", config.chunk_seconds.to_string());
    command.env("OVERLAP_SECONDS", config.overlap_seconds.to_string());
    command.env("FINAL_MIN_SECONDS", config.final_min_seconds.to_string());
    command.env("UTT_SPLIT_SECONDS", config.utt_split_seconds.to_string());
    command.env(
        "UPLOAD_RESPONSE_NUM_STREAMS",
        config.upload_response_num_streams.to_string(),
    );
    command.env(
        "UPLOAD_RESPONSE_SLOT_SIZE_KB",
        config.upload_response_slot_size_kb.to_string(),
    );
    command.env(
        "UPLOAD_RESPONSE_SLOTS_PER_STREAM",
        config.upload_response_slots_per_stream.to_string(),
    );
    command.env(
        "UPLOAD_RESPONSE_TIMEOUT_MS",
        config.upload_response_timeout_ms.to_string(),
    );
    command.env(
        "UPLOAD_RESPONSE_WATCH_POLL_MS",
        config.upload_response_watch_poll_ms.to_string(),
    );
    command.env(
        "UPLOAD_RESPONSE_WORKER_POLL_MS",
        config.upload_response_worker_poll_ms.to_string(),
    );
    command.env(
        "UPLOAD_RESPONSE_MAX_INFLIGHT",
        config.upload_response_max_inflight.to_string(),
    );
    command.env(
        "UPLOAD_RESPONSE_DISCOVERY_INTERVAL_MS",
        config.upload_response_discovery_interval_ms.to_string(),
    );
    command.env(
        "UPLOAD_RESPONSE_INSECURE_TLS",
        bool_env(config.upload_response_insecure_tls),
    );
    command.env(
        "UPLOAD_RESPONSE_WORKER_HEARTBEAT_INTERVAL_MS",
        config
            .upload_response_worker_heartbeat_interval_ms
            .to_string(),
    );
    command.env(
        "UPLOAD_RESPONSE_WORKER_TTL_MS",
        config.upload_response_worker_ttl_ms.to_string(),
    );

    if let Some(path) = &config.tls_cert_path {
        command.env("TLS_CERT_PATH", path);
    }
    if let Some(path) = &config.tls_key_path {
        command.env("TLS_KEY_PATH", path);
    }
    if let Some(path) = &config.vocab_path {
        command.env("ASR_VOCAB_PATH", path);
    }

    for (key, value) in role_env {
        command.env(key, value);
    }

    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn {name} service"))?;

    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(stream_output(name.clone(), "stdout", stdout));
    }
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(stream_output(name.clone(), "stderr", stderr));
    }

    Ok(Service { name, child })
}

async fn shutdown_services(services: &mut [Service]) {
    sleep(Duration::from_millis(250)).await;

    for service in services.iter_mut() {
        match service.child.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) => {
                let _ = service.child.start_kill();
            }
            Err(error) => {
                eprintln!(
                    "[orchestrator] failed to poll {} during shutdown: {error}",
                    service.name
                );
            }
        }
    }

    for service in services.iter_mut() {
        match service.child.wait().await {
            Ok(status) => {
                eprintln!("[orchestrator] {} stopped with {}", service.name, status);
            }
            Err(error) => {
                eprintln!(
                    "[orchestrator] failed waiting for {}: {error}",
                    service.name
                );
            }
        }
    }
}

async fn stream_output<R>(service: String, stream_name: &'static str, reader: R)
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut lines = BufReader::new(reader).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => eprintln!("[{service}:{stream_name}] {line}"),
            Ok(None) => break,
            Err(error) => {
                eprintln!("[orchestrator] failed to read {service}:{stream_name}: {error}");
                break;
            }
        }
    }
}

fn exe_name(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    }
}

fn validate_config(config: &Config) -> Result<()> {
    anyhow::ensure!(
        config.worker_count > 0,
        "--worker-count must be greater than 0"
    );
    worker_port(config, config.worker_count - 1)?;
    Ok(())
}

fn worker_port(config: &Config, index: usize) -> Result<u16> {
    let offset = u16::try_from(index).context("worker index does not fit in u16")?;
    config.worker_port.checked_add(offset).with_context(|| {
        format!(
            "--worker-count {} from --worker-port {} exceeds the u16 port range",
            config.worker_count, config.worker_port
        )
    })
}

fn worker_id(config: &Config, index: usize) -> String {
    if config.worker_count == 1 {
        config.upload_response_worker_id_prefix.clone()
    } else {
        format!("{}-{}", config.upload_response_worker_id_prefix, index + 1)
    }
}

fn worker_service_name(config: &Config, index: usize) -> String {
    if config.worker_count == 1 {
        "worker".to_string()
    } else {
        format!("worker-{}", index + 1)
    }
}

fn worker_urls(config: &Config) -> Result<Vec<String>> {
    (0..config.worker_count)
        .map(|index| worker_port(config, index).map(|port| format!("https://127.0.0.1:{port}")))
        .collect()
}

fn bool_env(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}
