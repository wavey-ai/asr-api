#[cfg(feature = "gpu-backend")]
pub mod asr;
pub mod chunking;
#[cfg(feature = "cohere-backend")]
pub mod cohere;
#[cfg(any(feature = "cohere-backend", feature = "cohere-mlx"))]
pub(crate) mod cohere_frontend;
#[cfg(feature = "cohere-mlx")]
pub mod cohere_mlx;
pub mod config;
#[cfg(feature = "cohere-backend")]
pub mod ctc_align;
#[cfg(feature = "audio-decoder")]
pub mod decoder;
pub mod deepgram;
pub mod ids;
pub mod ingress;
#[cfg(feature = "parakeet-backend")]
pub mod parakeet;
pub mod pcm;
pub mod processing;
pub mod protocol;
pub mod router;
pub(crate) mod timestamps;
#[cfg(feature = "gpu-backend")]
pub mod worker;

use anyhow::{Context, Result};
use clap::Parser;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};
use upload_response::{ResponseWatcher, UploadResponseRouter, UploadResponseService};
use web_service::{H2H3Server, Server, ServerBuilder};

#[cfg(feature = "gpu-backend")]
use crate::asr::AsrBackend;
#[cfg(any(feature = "gpu-backend", feature = "audio-decoder"))]
use crate::config::AppRole;
use crate::config::{AppConfig, LogFormat};
#[cfg(feature = "audio-decoder")]
use crate::decoder::DecoderState;
use crate::ingress::{ListenIngress, ListenIngressWebSocketHandler};
use crate::router::AppRouter;
#[cfg(feature = "gpu-backend")]
use crate::worker::WorkerState;

pub fn init_tracing(rust_log: &str, log_format: LogFormat) {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(rust_log));

    let _ = match log_format {
        LogFormat::Json => tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_span_events(FmtSpan::CLOSE)
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .json()
            .flatten_event(true)
            .with_current_span(true)
            .with_span_list(true)
            .try_init(),
        LogFormat::Pretty => tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_span_events(FmtSpan::CLOSE)
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .pretty()
            .try_init(),
        LogFormat::Compact => tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_span_events(FmtSpan::CLOSE)
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .compact()
            .try_init(),
    };
}

pub async fn run(config: AppConfig) -> Result<()> {
    config.validate()?;

    #[cfg(not(feature = "gpu-backend"))]
    if config.role.uses_asr_backend() {
        anyhow::bail!(
            "this asr-api build does not include the GPU backend; rebuild with the gpu-backend feature for role {:?}",
            config.role
        );
    }

    #[cfg(not(feature = "audio-decoder"))]
    if config.role.uses_audio_decoder() {
        anyhow::bail!(
            "this asr-api build does not include the audio decoder; rebuild with the audio-decoder feature for role {:?}",
            config.role
        );
    }

    let (cert_b64, key_b64) = config.tls_base64()?;
    let model_dir = if config.role.uses_asr_backend() {
        Some(config.model_dir()?.to_path_buf())
    } else {
        None
    };
    #[cfg(feature = "gpu-backend")]
    let model_provider = if config.role.uses_asr_backend() {
        Some(config.resolved_model_provider()?)
    } else {
        None
    };
    let upload_service = if config.role.exposes_upload_cache() {
        Some(Arc::new(UploadResponseService::new(
            config.upload_response_config(),
        )))
    } else {
        None
    };

    let _watcher_handle = upload_service.as_ref().map(|upload_service| {
        ResponseWatcher::new(upload_service.clone())
            .with_poll_interval_ms(config.upload_response_watch_poll_ms)
            .spawn()
    });

    #[cfg(feature = "gpu-backend")]
    let backend = if config.role.uses_asr_backend() {
        Some(Arc::new(AsrBackend::new(
            model_dir
                .as_ref()
                .expect("model dir already validated for backend role"),
            model_provider.expect("model provider already validated for backend role"),
            &config.device_ids,
            config.onnx_sessions,
            config.cohere_max_new_tokens,
        )?))
    } else {
        None
    };

    #[cfg(feature = "gpu-backend")]
    let worker_state = backend.map(|backend| Arc::new(WorkerState::new(config.clone(), backend)));
    #[cfg(feature = "audio-decoder")]
    let decoder_state = if config.role.uses_audio_decoder() {
        Some(Arc::new(DecoderState::new(config.clone())))
    } else {
        None
    };

    #[cfg(any(feature = "gpu-backend", feature = "audio-decoder"))]
    let _worker_handle = {
        #[allow(unused_mut)]
        let mut handle = None;
        #[cfg(feature = "gpu-backend")]
        if matches!(config.role, AppRole::Worker) {
            handle = Some(
                worker_state
                    .as_ref()
                    .expect("worker role must have worker state")
                    .clone()
                    .spawn_remote_cache_worker(),
            );
        }
        #[cfg(feature = "audio-decoder")]
        if matches!(config.role, AppRole::Decoder) {
            handle = Some(
                decoder_state
                    .as_ref()
                    .expect("decoder role must have decoder state")
                    .clone()
                    .spawn_remote_cache_worker(),
            );
        }
        handle
    };

    let upload_router = upload_service
        .as_ref()
        .map(|upload_service| Arc::new(UploadResponseRouter::new(upload_service.clone())));
    let listen_ingress = upload_service
        .as_ref()
        .map(|upload_service| Arc::new(ListenIngress::new(config.clone(), upload_service.clone())));
    let listen_ws = listen_ingress
        .as_ref()
        .map(|ingress| Arc::new(ListenIngressWebSocketHandler::new(ingress.clone())));

    let router = Box::new(AppRouter::new(
        config.clone(),
        upload_router,
        listen_ingress,
        listen_ws.clone(),
    ));

    let enable_websocket = listen_ws.is_some();

    let server = H2H3Server::builder()
        .with_tls(cert_b64, key_b64)
        .with_port(config.port)
        .enable_h2(true)
        .enable_h3(config.enable_h3)
        .enable_websocket(enable_websocket)
        .with_router(router)
        .build()
        .context("failed to build server")?;

    let handle = server.start().await.context("failed to start server")?;
    let _ = handle.ready_rx.await;
    info!(
        role = ?config.role,
        port = config.port,
        enable_h3 = config.enable_h3,
        model_dir = %model_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
        model_provider = %config.default_model_provider().default_model_name(),
        device_ids = ?config.device_ids,
        onnx_sessions = config.onnx_sessions,
        chunk_seconds = config.chunk_seconds,
        overlap_seconds = config.overlap_seconds,
        upload_response_num_streams = config.upload_response_num_streams,
        upload_response_slot_size_kb = config.upload_response_slot_size_kb,
        upload_response_worker_id = %config.upload_response_worker_id,
        upload_response_discovery_dns = ?config.upload_response_discovery_dns,
        upload_response_ingress_urls = ?config.upload_response_ingress_urls,
        "asr-api ready"
    );

    tokio::signal::ctrl_c().await?;
    info!("shutdown signal received");
    let _ = handle.shutdown_tx.send(());
    let _ = handle.finished_rx.await;
    Ok(())
}

pub async fn run_from_env() -> Result<()> {
    dotenvy::dotenv().ok();
    let config = AppConfig::parse();
    init_tracing(&config.rust_log, config.log_format);
    run(config).await
}
