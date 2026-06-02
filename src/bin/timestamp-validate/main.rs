use anyhow::{Context, Result};
use asr_api::asr::{AsrBackend, WindowTranscription};
use asr_api::chunking::TimedWord;
use asr_api::config::{AsrModelProvider, ASR_SAMPLE_RATE};
use bytes::Bytes;
use clap::Parser;
use soundkit::audio_pipeline::audio_to_mono_f32;
use soundkit_decoder::{DecodeError, DecodeOptions, DecodePipeline};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    cohere_model_dir: PathBuf,
    #[arg(long)]
    parakeet_model_dir: PathBuf,
    #[arg(long)]
    audio_path: PathBuf,
    #[arg(long, default_value = "0")]
    device_ids: String,
    #[arg(long, default_value_t = 1)]
    onnx_sessions: usize,
    #[arg(long, default_value_t = 384)]
    max_new_tokens: usize,
    #[arg(long)]
    show_words: bool,
}

#[derive(Debug)]
struct MatchedWord<'a> {
    cohere: &'a TimedWord,
    parakeet: &'a TimedWord,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let samples = decode_audio(&args.audio_path).await?;
    let device_ids = parse_device_ids(&args.device_ids)?;
    let audio_seconds = samples.len() as f64 / f64::from(ASR_SAMPLE_RATE);

    let cohere = AsrBackend::new(
        &args.cohere_model_dir,
        AsrModelProvider::Cohere,
        &device_ids,
        args.onnx_sessions,
        args.max_new_tokens,
    )
    .context("failed to initialize Cohere backend")?;
    let parakeet = AsrBackend::new(
        &args.parakeet_model_dir,
        AsrModelProvider::Parakeet,
        &device_ids,
        args.onnx_sessions,
        args.max_new_tokens,
    )
    .context("failed to initialize Parakeet backend")?;

    let cohere_started = Instant::now();
    let cohere_result = cohere
        .transcribe_window(samples.clone(), 0)
        .await
        .context("Cohere transcription failed")?;
    let cohere_elapsed = cohere_started.elapsed().as_secs_f64();

    let parakeet_started = Instant::now();
    let parakeet_result = parakeet
        .transcribe_window(samples, 1)
        .await
        .context("Parakeet transcription failed")?;
    let parakeet_elapsed = parakeet_started.elapsed().as_secs_f64();

    print_summary("cohere", &cohere_result, cohere_elapsed, audio_seconds);
    print_summary(
        "parakeet",
        &parakeet_result,
        parakeet_elapsed,
        audio_seconds,
    );
    print_metrics(
        &align_words(&cohere_result.words, &parakeet_result.words),
        cohere_result.words.len(),
        parakeet_result.words.len(),
        args.show_words,
    );

    std::mem::forget(cohere);
    std::mem::forget(parakeet);
    std::process::exit(0);
}

async fn decode_audio(path: &PathBuf) -> Result<Vec<f32>> {
    let audio_bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
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
    Ok(samples)
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

fn parse_device_ids(value: &str) -> Result<Vec<usize>> {
    value
        .split(',')
        .filter_map(|part| {
            let trimmed = part.trim();
            (!trimmed.is_empty()).then_some(trimmed)
        })
        .map(|part| {
            part.parse::<usize>()
                .with_context(|| format!("invalid device id: {part}"))
        })
        .collect()
}

fn print_summary(label: &str, result: &WindowTranscription, elapsed_s: f64, audio_seconds: f64) {
    println!("{label}_text={}", result.text);
    println!("{label}_words={}", result.words.len());
    println!("{label}_decode_ms={:.2}", elapsed_s * 1000.0);
    println!("{label}_rtfx={:.2}", audio_seconds / elapsed_s);
}

fn align_words<'a>(cohere: &'a [TimedWord], parakeet: &'a [TimedWord]) -> Vec<MatchedWord<'a>> {
    let left = cohere
        .iter()
        .map(|word| normalize(&word.word))
        .collect::<Vec<_>>();
    let right = parakeet
        .iter()
        .map(|word| normalize(&word.word))
        .collect::<Vec<_>>();
    let rows = left.len() + 1;
    let cols = right.len() + 1;
    let mut dp = vec![0usize; rows * cols];

    for i in (0..left.len()).rev() {
        for j in (0..right.len()).rev() {
            let value = if !left[i].is_empty() && left[i] == right[j] {
                dp[(i + 1) * cols + j + 1] + 1
            } else {
                dp[(i + 1) * cols + j].max(dp[i * cols + j + 1])
            };
            dp[i * cols + j] = value;
        }
    }

    let mut matches = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left.len() && j < right.len() {
        if !left[i].is_empty() && left[i] == right[j] {
            matches.push(MatchedWord {
                cohere: &cohere[i],
                parakeet: &parakeet[j],
            });
            i += 1;
            j += 1;
        } else if dp[(i + 1) * cols + j] >= dp[i * cols + j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }

    matches
}

fn print_metrics(
    matches: &[MatchedWord<'_>],
    cohere_word_count: usize,
    parakeet_word_count: usize,
    show_words: bool,
) {
    println!("matched_words={}", matches.len());
    println!("cohere_word_count={cohere_word_count}");
    println!("parakeet_word_count={parakeet_word_count}");

    if matches.is_empty() {
        return;
    }

    let mut start_deltas = Vec::with_capacity(matches.len());
    let mut end_deltas = Vec::with_capacity(matches.len());
    let mut mid_deltas = Vec::with_capacity(matches.len());

    for matched in matches {
        let cohere_mid = midpoint_ms(matched.cohere);
        let parakeet_mid = midpoint_ms(matched.parakeet);
        start_deltas.push(abs_diff_ms(
            matched.cohere.start_ms,
            matched.parakeet.start_ms,
        ));
        end_deltas.push(abs_diff_ms(matched.cohere.end_ms, matched.parakeet.end_ms));
        mid_deltas.push(abs_diff_ms(cohere_mid, parakeet_mid));

        if show_words {
            println!(
                "word={} cohere={}..{} parakeet={}..{} mid_delta_ms={}",
                matched.cohere.word,
                matched.cohere.start_ms,
                matched.cohere.end_ms,
                matched.parakeet.start_ms,
                matched.parakeet.end_ms,
                abs_diff_ms(cohere_mid, parakeet_mid)
            );
        }
    }

    mid_deltas.sort_unstable();
    println!("mean_abs_start_delta_ms={:.2}", mean(&start_deltas));
    println!("mean_abs_end_delta_ms={:.2}", mean(&end_deltas));
    println!("mean_abs_mid_delta_ms={:.2}", mean(&mid_deltas));
    println!("p50_abs_mid_delta_ms={}", percentile(&mid_deltas, 50.0));
    println!("p95_abs_mid_delta_ms={}", percentile(&mid_deltas, 95.0));
}

fn normalize(word: &str) -> String {
    word.chars()
        .filter(|ch| ch.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn midpoint_ms(word: &TimedWord) -> u32 {
    word.start_ms + ((word.end_ms.saturating_sub(word.start_ms)) / 2)
}

fn abs_diff_ms(left: u32, right: u32) -> u32 {
    left.max(right) - left.min(right)
}

fn mean(values: &[u32]) -> f64 {
    values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64
}

fn percentile(sorted_values: &[u32], percentile: f64) -> u32 {
    if sorted_values.is_empty() {
        return 0;
    }
    let rank = ((percentile / 100.0) * (sorted_values.len().saturating_sub(1) as f64)).round();
    sorted_values[rank as usize]
}
