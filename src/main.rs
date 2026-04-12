use anyhow::{Context, Result};
use clap::Parser;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::EnvFilter;
use web_service::{H2H3Server, Server, ServerBuilder};

use transcriber::config::AppConfig;
use transcriber::model::ModelPool;
use transcriber::router::AppRouter;
use transcriber::worker::WorkerState;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let config = AppConfig::parse();
    config.validate()?;

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.rust_log)),
        )
        .init();

    let (cert_b64, key_b64) = config.tls_base64()?;
    let model_pool = Arc::new(ModelPool::new(&config.model_dir, config.model_instances)?);
    let worker_state = Arc::new(WorkerState::new(config.clone(), model_pool));
    let router = Box::new(AppRouter::new(config.clone(), worker_state));

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
        model_dir = %config.model_dir.display(),
        chunk_seconds = config.chunk_seconds,
        overlap_seconds = config.overlap_seconds,
        "transcriber ready"
    );

    tokio::signal::ctrl_c().await?;
    info!("shutdown signal received");
    let _ = handle.shutdown_tx.send(());
    let _ = handle.finished_rx.await;
    Ok(())
}
