use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    transcriber::run_from_env().await
}
