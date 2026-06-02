use anyhow::{Context, Result};
use asr_api::asr::AsrBackend;
use asr_api::config::AsrModelProvider;
use bytes::Bytes;
use clap::Parser;
use soundkit::audio_pipeline::audio_to_mono_f32;
use soundkit_decoder::{DecodeError, DecodeOptions, DecodePipeline};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const ASR_SAMPLE_RATE: u32 = 16_000;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    audio_path: PathBuf,
    #[arg(long, value_enum, default_value_t = AsrModelProvider::Cohere)]
    model_provider: AsrModelProvider,
    #[arg(long, default_value = "0")]
    device_ids: String,
    #[arg(long, default_value_t = 1)]
    onnx_sessions: usize,
    #[arg(long, default_value_t = 384)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    #[arg(long, default_value_t = 1)]
    repeat: usize,
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

    let audio_seconds = samples.len() as f64 / f64::from(ASR_SAMPLE_RATE);

    let init_started = Instant::now();
    let backend = AsrBackend::new(
        &args.model_dir,
        args.model_provider,
        &device_ids,
        args.onnx_sessions,
        args.max_new_tokens,
    )?;
    let init_elapsed = init_started.elapsed();

    for idx in 0..args.warmup {
        let started = Instant::now();
        let result = backend
            .transcribe_window(samples.clone(), idx as u32)
            .await?;
        eprintln!(
            "warmup={} decode_ms={:.2} rtfx={:.2} text_len={}",
            idx + 1,
            started.elapsed().as_secs_f64() * 1000.0,
            audio_seconds / started.elapsed().as_secs_f64(),
            result.text.len()
        );
    }

    let mut decode_times = Vec::with_capacity(args.repeat);
    let mut last_result = None;
    for idx in 0..args.repeat {
        let started = Instant::now();
        let result = backend
            .transcribe_window(samples.clone(), (args.warmup + idx) as u32)
            .await?;
        let elapsed = started.elapsed();
        eprintln!(
            "repeat={} decode_ms={:.2} rtfx={:.2} text_len={}",
            idx + 1,
            elapsed.as_secs_f64() * 1000.0,
            audio_seconds / elapsed.as_secs_f64(),
            result.text.len()
        );
        decode_times.push(elapsed.as_secs_f64());
        last_result = Some(result);
    }

    let result = last_result.context("repeat must be at least 1")?;
    let mean_decode_s = decode_times.iter().sum::<f64>() / decode_times.len() as f64;

    println!("text={}", result.text);
    println!("text_len={}", result.text.len());
    println!("words={}", result.words.len());
    println!("audio_seconds={audio_seconds:.3}");
    println!("init_ms={:.2}", init_elapsed.as_secs_f64() * 1000.0);
    println!("mean_decode_ms={:.2}", mean_decode_s * 1000.0);
    println!("mean_rtfx={:.2}", audio_seconds / mean_decode_s);
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
