pub mod asr;
pub mod chunking;
pub mod config;
pub mod deepgram;
pub mod router;
pub mod worker;

use anyhow::{Context, Result};
use clap::Parser;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::EnvFilter;
use upload_response::{ResponseWatcher, UploadResponseRouter, UploadResponseService};
use web_service::{H2H3Server, Server, ServerBuilder};

use crate::asr::AsrBackend;
use crate::config::AppConfig;
use crate::router::AppRouter;
use crate::worker::WorkerState;

pub fn init_tracing(rust_log: &str) {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(rust_log)),
        )
        .try_init();
}

pub async fn run(config: AppConfig) -> Result<()> {
    config.validate()?;

    let model_dir = config.model_dir.clone();
    let vocab_path = config.resolve_vocab_path()?;
    let (cert_b64, key_b64) = config.tls_base64()?;
    let backend = Arc::new(AsrBackend::new(
        &model_dir,
        &vocab_path,
        &config.device_ids,
        config.torch_sessions,
        config.onnx_sessions,
    )?);
    let worker_state = Arc::new(WorkerState::new(config.clone(), backend));
    let upload_service = Arc::new(UploadResponseService::new(config.upload_response_config()));
    let _watcher_handle = ResponseWatcher::new(upload_service.clone())
        .with_poll_interval_ms(config.upload_response_watch_poll_ms)
        .spawn();
    let _worker_handle = worker_state.clone().spawn_cache_worker(upload_service.clone());

    let upload_router = Arc::new(UploadResponseRouter::new(upload_service));
    let router = Box::new(AppRouter::new(upload_router));

    let server = H2H3Server::builder()
        .with_tls(cert_b64, key_b64)
        .with_port(config.port)
        .enable_h2(true)
        .enable_h3(config.enable_h3)
        .enable_websocket(false)
        .with_router(router)
        .build()
        .context("failed to build server")?;

    let handle = server.start().await.context("failed to start server")?;
    let _ = handle.ready_rx.await;
    info!(
        port = config.port,
        enable_h3 = config.enable_h3,
        model_dir = %model_dir.display(),
        vocab_path = %vocab_path.display(),
        device_ids = ?config.device_ids,
        torch_sessions = config.torch_sessions,
        onnx_sessions = config.onnx_sessions,
        chunk_seconds = config.chunk_seconds,
        overlap_seconds = config.overlap_seconds,
        upload_response_num_streams = config.upload_response_num_streams,
        upload_response_slot_size_kb = config.upload_response_slot_size_kb,
        upload_response_worker_id = %config.upload_response_worker_id,
        "transcriber ready"
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
    init_tracing(&config.rust_log);
    run(config).await
}
