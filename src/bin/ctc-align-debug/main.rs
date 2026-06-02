use anyhow::{Context, Result};
use asr_api::chunking::TimedWord;
use asr_api::config::ASR_SAMPLE_RATE;
use asr_api::ctc_align::ParakeetCtcAligner;
use bytes::Bytes;
use clap::Parser;
use serde::Deserialize;
use soundkit::audio_pipeline::audio_to_mono_f32;
use soundkit_decoder::{DecodeError, DecodeOptions, DecodePipeline};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    audio_path: PathBuf,
    #[arg(long)]
    text: Option<String>,
    #[arg(long)]
    text_file: Option<PathBuf>,
    #[arg(long)]
    coherex_transcripts_json: Option<PathBuf>,
    #[arg(long)]
    coherex_alignment_json: Option<PathBuf>,
    #[arg(long)]
    baseline_key: Option<String>,
    #[arg(long, default_value = "0")]
    device_ids: String,
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    #[arg(long, default_value_t = 1)]
    repeat: usize,
    #[arg(long)]
    show_words: bool,
}

#[derive(Debug, Deserialize)]
struct CohereXTranscript {
    word_sequence: String,
}

#[derive(Debug, Deserialize)]
struct CohereXAlignment {
    word_segments: Vec<CohereXWord>,
}

#[derive(Debug, Deserialize)]
struct CohereXWord {
    word: String,
    start: f64,
    end: f64,
}

#[derive(Debug)]
struct MatchedWord<'a> {
    ctc: &'a TimedWord,
    baseline: &'a CohereXWord,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let key = args
        .baseline_key
        .clone()
        .or_else(|| {
            args.audio_path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .context("failed to infer baseline key from --audio-path")?;
    let text = load_text(&args, &key)?;
    let samples = decode_audio(&args.audio_path).await?;
    let audio_seconds = samples.len() as f64 / f64::from(ASR_SAMPLE_RATE);
    let device_ids = parse_device_ids(&args.device_ids)?;

    let init_started = Instant::now();
    let aligner =
        ParakeetCtcAligner::new(&device_ids).context("failed to initialize CTC aligner")?;
    let init_elapsed = init_started.elapsed().as_secs_f64();

    for idx in 0..args.warmup {
        let started = Instant::now();
        let words = aligner.align(&samples, &text)?;
        eprintln!(
            "warmup={} align_ms={:.2} rtfx={:.2} words={}",
            idx + 1,
            started.elapsed().as_secs_f64() * 1000.0,
            audio_seconds / started.elapsed().as_secs_f64(),
            words.len()
        );
    }

    let mut align_times = Vec::with_capacity(args.repeat);
    let mut last_words = Vec::new();
    for idx in 0..args.repeat {
        let started = Instant::now();
        let words = aligner.align(&samples, &text)?;
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!(
            "repeat={} align_ms={:.2} rtfx={:.2} words={}",
            idx + 1,
            elapsed * 1000.0,
            audio_seconds / elapsed,
            words.len()
        );
        align_times.push(elapsed);
        last_words = words;
    }

    let mean_align_s = align_times.iter().sum::<f64>() / align_times.len() as f64;
    println!("baseline_key={key}");
    println!("text_len={}", text.len());
    println!("audio_seconds={audio_seconds:.3}");
    println!("init_ms={:.2}", init_elapsed * 1000.0);
    println!("words={}", last_words.len());
    println!("mean_align_ms={:.2}", mean_align_s * 1000.0);
    println!("mean_rtfx={:.2}", audio_seconds / mean_align_s);

    if let Some(path) = &args.coherex_alignment_json {
        let baseline = load_alignment(path, &key)?;
        print_metrics(&last_words, &baseline.word_segments, args.show_words);
    } else if args.show_words {
        for word in &last_words {
            println!("word={} ctc={}..{}", word.word, word.start_ms, word.end_ms);
        }
    }

    std::mem::forget(aligner);
    std::process::exit(0);
}

fn load_text(args: &Args, key: &str) -> Result<String> {
    if let Some(text) = &args.text {
        return Ok(text.clone());
    }
    if let Some(path) = &args.text_file {
        return std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()));
    }
    if let Some(path) = &args.coherex_transcripts_json {
        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let root: serde_json::Map<String, serde_json::Value> = serde_json::from_reader(file)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let transcript: CohereXTranscript = serde_json::from_value(
            root.get(key)
                .cloned()
                .with_context(|| format!("missing CohereX transcript key {key:?}"))?,
        )
        .with_context(|| format!("failed to parse CohereX transcript for {key:?}"))?;
        return Ok(transcript.word_sequence);
    }
    anyhow::bail!("provide --text, --text-file, or --coherex-transcripts-json")
}

fn load_alignment(path: &Path, key: &str) -> Result<CohereXAlignment> {
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let root: serde_json::Map<String, serde_json::Value> = serde_json::from_reader(file)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    serde_json::from_value(
        root.get(key)
            .cloned()
            .with_context(|| format!("missing CohereX alignment key {key:?}"))?,
    )
    .with_context(|| format!("failed to parse CohereX alignment for {key:?}"))
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

fn print_metrics(ctc: &[TimedWord], baseline: &[CohereXWord], show_words: bool) {
    println!("baseline_words={}", baseline.len());
    let matches = align_words(ctc, baseline);
    println!("matched_words={}", matches.len());
    if matches.is_empty() {
        return;
    }

    let mut start_deltas = Vec::with_capacity(matches.len());
    let mut end_deltas = Vec::with_capacity(matches.len());
    let mut mid_deltas = Vec::with_capacity(matches.len());
    for matched in matches {
        let baseline_start = seconds_to_ms(matched.baseline.start);
        let baseline_end = seconds_to_ms(matched.baseline.end);
        let ctc_mid = midpoint_ms(matched.ctc.start_ms, matched.ctc.end_ms);
        let baseline_mid = midpoint_ms(baseline_start, baseline_end);
        start_deltas.push(abs_diff_ms(matched.ctc.start_ms, baseline_start));
        end_deltas.push(abs_diff_ms(matched.ctc.end_ms, baseline_end));
        mid_deltas.push(abs_diff_ms(ctc_mid, baseline_mid));
        if show_words {
            println!(
                "word={} ctc={}..{} coherex={}..{} mid_delta_ms={}",
                matched.ctc.word,
                matched.ctc.start_ms,
                matched.ctc.end_ms,
                baseline_start,
                baseline_end,
                abs_diff_ms(ctc_mid, baseline_mid)
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

fn align_words<'a>(ctc: &'a [TimedWord], baseline: &'a [CohereXWord]) -> Vec<MatchedWord<'a>> {
    let left = ctc
        .iter()
        .map(|word| normalize(&word.word))
        .collect::<Vec<_>>();
    let right = baseline
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
                ctc: &ctc[i],
                baseline: &baseline[j],
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

fn normalize(word: &str) -> String {
    word.chars()
        .filter(|ch| ch.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn seconds_to_ms(value: f64) -> u32 {
    (value * 1000.0).round().clamp(0.0, f64::from(u32::MAX)) as u32
}

fn midpoint_ms(start_ms: u32, end_ms: u32) -> u32 {
    start_ms + ((end_ms.saturating_sub(start_ms)) / 2)
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
