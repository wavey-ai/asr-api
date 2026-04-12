#[cfg(feature = "gpu-backend")]
pub mod asr;
pub mod chunking;
pub mod config;
pub mod deepgram;
pub mod ingress;
pub mod pcm;
pub mod protocol;
pub mod router;
#[cfg(feature = "gpu-backend")]
pub mod worker;

use anyhow::{Context, Result};
use clap::Parser;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::EnvFilter;
use upload_response::{ResponseWatcher, UploadResponseRouter, UploadResponseService};
use web_service::{H2H3Server, Server, ServerBuilder};

#[cfg(feature = "gpu-backend")]
use crate::asr::AsrBackend;
use crate::config::AppConfig;
#[cfg(feature = "gpu-backend")]
use crate::config::AppRole;
use crate::ingress::{ListenIngress, ListenIngressWebSocketHandler};
use crate::router::AppRouter;
#[cfg(feature = "gpu-backend")]
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

    #[cfg(not(feature = "gpu-backend"))]
    if config.role.uses_asr_backend() {
        anyhow::bail!(
            "this asr-api build does not include the GPU backend; use the worker image for role {:?}",
            config.role
        );
    }

    let (cert_b64, key_b64) = config.tls_base64()?;
    let model_dir = if config.role.uses_asr_backend() {
        Some(config.model_dir()?.to_path_buf())
    } else {
        None
    };
    let vocab_path = if config.role.uses_asr_backend() {
        Some(config.resolve_vocab_path()?)
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
            vocab_path
                .as_ref()
                .expect("vocab path already validated for backend role"),
            &config.device_ids,
            config.torch_sessions,
            config.onnx_sessions,
        )?))
    } else {
        None
    };

    #[cfg(feature = "gpu-backend")]
    let worker_state = backend.map(|backend| Arc::new(WorkerState::new(config.clone(), backend)));

    #[cfg(feature = "gpu-backend")]
    let _worker_handle = match config.role {
        AppRole::Monolith => Some(
            worker_state
                .as_ref()
                .expect("monolith role must have worker state")
                .clone()
                .spawn_cache_worker(
                    upload_service
                        .as_ref()
                        .expect("monolith role must have upload cache")
                        .clone(),
                ),
        ),
        AppRole::Worker => Some(
            worker_state
                .as_ref()
                .expect("worker role must have worker state")
                .clone()
                .spawn_remote_cache_worker(),
        ),
        AppRole::Ingress => None,
    };

    let upload_router = upload_service
        .as_ref()
        .map(|upload_service| Arc::new(UploadResponseRouter::new(upload_service.clone())));
    let listen_ingress = upload_service
        .as_ref()
        .filter(|_| config.role.serves_listen())
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
        vocab_path = %vocab_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
        device_ids = ?config.device_ids,
        torch_sessions = config.torch_sessions,
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
    init_tracing(&config.rust_log);
    run(config).await
}
