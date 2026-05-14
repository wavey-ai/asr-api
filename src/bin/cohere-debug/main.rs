use anyhow::{Context, Result};
use asr_api::cohere::CohereBackend;
use bytes::Bytes;
use clap::Parser;
use soundkit::audio_pipeline::audio_to_mono_f32;
use soundkit_decoder::{DecodeError, DecodeOptions, DecodePipeline};
use std::fs;
use std::path::PathBuf;

const ASR_SAMPLE_RATE: u32 = 16_000;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    audio_path: PathBuf,
    #[arg(long, default_value = "0")]
    device_ids: String,
    #[arg(long, default_value_t = 1)]
    onnx_sessions: usize,
    #[arg(long, default_value_t = 384)]
    max_new_tokens: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let audio_bytes = fs::read(&args.audio_path)
        .with_context(|| format!("failed to read {}", args.audio_path.display()))?;
    let mut decoder = DecodePipeline::spawn_with_options(DecodeOptions {
        output_bits_per_sample: Some(16),
        output_sample_rate: Some(ASR_SAMPLE_RATE),
        output_channels: Some(1),
    });
    send_all(&mut decoder, Bytes::from(audio_bytes)).await?;
    send_all(&mut decoder, Bytes::new()).await?;

    let mut samples = Vec::new();
    while let Some(output) = decoder.recv() {
        let audio = output.map_err(|error| anyhow::anyhow!("decode failed: {error}"))?;
        let mono = audio_to_mono_f32(&audio).map_err(anyhow::Error::msg)?;
        samples.extend_from_slice(&mono);
    }

    let device_ids = args
        .device_ids
        .split(',')
        .filter_map(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then_some(trimmed)
        })
        .map(|value| {
            value
                .parse::<usize>()
                .with_context(|| format!("invalid device id: {value}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let backend = CohereBackend::new(
        &args.model_dir,
        &device_ids,
        args.onnx_sessions,
        args.max_new_tokens,
    )?;
    let result = backend.transcribe_window(samples, 0).await?;

    println!("text={}", result.text);
    println!("text_len={}", result.text.len());
    println!("words={}", result.words.len());
    std::mem::forget(backend);
    std::process::exit(0);
}

async fn send_all(decoder: &mut soundkit_decoder::DecodePipelineHandle, data: Bytes) -> Result<()> {
    loop {
        match decoder.send(data.clone()) {
            Ok(()) => return Ok(()),
            Err(DecodeError::InputBufferFull) => tokio::task::yield_now().await,
            Err(error) => return Err(anyhow::anyhow!("decoder send failed: {error}")),
        }
    }
}
