use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    asr_api::run_from_env().await
}
