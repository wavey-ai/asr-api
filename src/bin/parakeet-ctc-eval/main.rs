use anyhow::{Context, Result};
use asr_api::chunking::TimedWord;
use asr_api::config::ASR_SAMPLE_RATE;
use asr_api::ctc_align::ParakeetCtcAligner;
use bytes::Bytes;
use clap::Parser;
use serde::{Deserialize, Serialize};
use soundkit::audio_pipeline::audio_to_mono_f32;
use soundkit_decoder::{DecodeError, DecodeOptions, DecodePipeline};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    corpus_dir: PathBuf,
    #[arg(long, default_value = "0")]
    device_ids: String,
    #[arg(long, default_value_t = 5)]
    warmup_utterances: usize,
    #[arg(long, default_value_t = 1)]
    repeat: usize,
    #[arg(long)]
    output_json: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct ReferenceAlignment {
    word_segments: Vec<ReferenceWord>,
}

#[derive(Debug, Clone, Deserialize)]
struct ReferenceWord {
    word: String,
    start: f64,
    end: f64,
}

#[derive(Debug, Serialize)]
struct EvalOutput {
    result: EvalResult,
    summary: Vec<UtteranceSummary>,
}

#[derive(Debug, Serialize)]
struct EvalResult {
    utterances: usize,
    failures: Vec<String>,
    audio_seconds: f64,
    reference_words: usize,
    hypothesis_words: usize,
    matched_words: usize,
    word_error_rate: f64,
    anchored_words: usize,
    anchored_mean_start_ms: f64,
    anchored_mean_end_ms: f64,
    anchored_mean_mid_ms: f64,
    anchored_p50_mid_ms: u32,
    anchored_p90_mid_ms: u32,
    anchored_p95_mid_ms: u32,
    anchored_p99_mid_ms: u32,
    direct_calibrated: CalibratedMetrics,
    anchored_calibrated: CalibratedMetrics,
    total_mean_decode_ms: f64,
    rtfx: f64,
    mean_start_ms: f64,
    mean_end_ms: f64,
    mean_mid_ms: f64,
    p50_mid_ms: u32,
    p90_mid_ms: u32,
    p95_mid_ms: u32,
    p99_mid_ms: u32,
    per_utterance_mean_decode_ms: f64,
    per_utterance_p50_decode_ms: f64,
    per_utterance_p95_decode_ms: f64,
}

#[derive(Debug, Serialize)]
struct UtteranceSummary {
    key: String,
    duration: f64,
    reference_words: usize,
    hypothesis_words: usize,
    matched_words: usize,
    word_error_rate: f64,
    anchored_mean_mid_ms: f64,
    mean_decode_ms: f64,
    text: String,
}

#[derive(Debug, Serialize)]
struct CalibratedMetrics {
    start_offset_ms: i32,
    end_offset_ms: i32,
    mean_start_ms: f64,
    mean_end_ms: f64,
    mean_mid_ms: f64,
    p50_mid_ms: u32,
    p90_mid_ms: u32,
    p95_mid_ms: u32,
    p99_mid_ms: u32,
}

#[derive(Debug)]
struct MatchedWord<'a> {
    hypothesis: &'a TimedWord,
    reference: &'a ReferenceWord,
}

#[derive(Debug, Clone, Copy)]
struct SignedBoundaryPair {
    start_ms: i32,
    end_ms: i32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let alignments = load_alignments(&args.corpus_dir.join("alignment.json"))?;
    let device_ids = parse_device_ids(&args.device_ids)?;
    let repeat = args.repeat.max(1);

    let init_started = Instant::now();
    let aligner =
        ParakeetCtcAligner::new(&device_ids).context("failed to initialize Parakeet CTC")?;
    eprintln!(
        "parakeet_ctc_init_ms={:.2}",
        init_started.elapsed().as_secs_f64() * 1000.0
    );

    for key in alignments.keys().take(args.warmup_utterances) {
        let samples = decode_audio(&args.corpus_dir.join("audio").join(key)).await?;
        let started = Instant::now();
        let result = aligner.transcribe(&samples)?;
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!(
            "warmup key={} decode_ms={:.2} words={}",
            key,
            elapsed * 1000.0,
            result.words.len()
        );
    }

    let mut summaries = Vec::new();
    let mut failures = Vec::new();
    let mut decode_times = Vec::new();
    let mut all_start_deltas = Vec::new();
    let mut all_end_deltas = Vec::new();
    let mut all_mid_deltas = Vec::new();
    let mut all_signed_pairs = Vec::new();
    let mut anchored_start_deltas = Vec::new();
    let mut anchored_end_deltas = Vec::new();
    let mut anchored_mid_deltas = Vec::new();
    let mut anchored_signed_pairs = Vec::new();
    let mut total_audio_seconds = 0.0;
    let mut total_reference_words = 0usize;
    let mut total_hypothesis_words = 0usize;
    let mut total_matched_words = 0usize;
    let mut total_anchored_words = 0usize;
    let mut total_edit_distance = 0usize;

    for (seq, (key, reference)) in alignments.iter().enumerate() {
        let audio_path = args.corpus_dir.join("audio").join(key);
        let samples = match decode_audio(&audio_path).await {
            Ok(samples) => samples,
            Err(error) => {
                failures.push(format!("{key}: decode failed: {error:?}"));
                continue;
            }
        };
        let duration = samples.len() as f64 / f64::from(ASR_SAMPLE_RATE);
        total_audio_seconds += duration;
        total_reference_words += reference.word_segments.len();

        let mut last_result = None;
        let mut utterance_times = Vec::with_capacity(repeat);
        for _ in 0..repeat {
            let started = Instant::now();
            match aligner.transcribe(&samples) {
                Ok(result) => {
                    let elapsed = started.elapsed().as_secs_f64();
                    utterance_times.push(elapsed);
                    decode_times.push(elapsed * 1000.0);
                    last_result = Some(result);
                }
                Err(error) => {
                    failures.push(format!("{key}: CTC transcribe failed: {error:?}"));
                    break;
                }
            }
        }

        let Some(result) = last_result else {
            continue;
        };
        let mean_decode_ms =
            utterance_times.iter().sum::<f64>() * 1000.0 / utterance_times.len().max(1) as f64;
        let matches = align_words(&result.words, &reference.word_segments);
        let hyp_norm = result
            .words
            .iter()
            .map(|word| normalize_word(&word.word))
            .filter(|word| !word.is_empty())
            .collect::<Vec<_>>();
        let ref_norm = reference
            .word_segments
            .iter()
            .map(|word| normalize_word(&word.word))
            .filter(|word| !word.is_empty())
            .collect::<Vec<_>>();
        let edit_distance = edit_distance(&hyp_norm, &ref_norm);

        for matched in &matches {
            let ref_start = seconds_to_ms(matched.reference.start);
            let ref_end = seconds_to_ms(matched.reference.end);
            let hyp_mid = midpoint_ms(matched.hypothesis.start_ms, matched.hypothesis.end_ms);
            let ref_mid = midpoint_ms(ref_start, ref_end);
            all_start_deltas.push(abs_diff_ms(matched.hypothesis.start_ms, ref_start));
            all_end_deltas.push(abs_diff_ms(matched.hypothesis.end_ms, ref_end));
            all_mid_deltas.push(abs_diff_ms(hyp_mid, ref_mid));
            all_signed_pairs.push(SignedBoundaryPair {
                start_ms: signed_diff_ms(matched.hypothesis.start_ms, ref_start),
                end_ms: signed_diff_ms(matched.hypothesis.end_ms, ref_end),
            });
        }

        let anchored_words = timestamp_reference_from_hypothesis(
            &result.words,
            &reference.word_segments,
            samples.len(),
        );
        let mut utterance_anchor_mid_deltas = Vec::new();
        for (anchored, reference_word) in anchored_words.iter().zip(&reference.word_segments) {
            let ref_start = seconds_to_ms(reference_word.start);
            let ref_end = seconds_to_ms(reference_word.end);
            let anchored_mid = midpoint_ms(anchored.start_ms, anchored.end_ms);
            let ref_mid = midpoint_ms(ref_start, ref_end);
            anchored_start_deltas.push(abs_diff_ms(anchored.start_ms, ref_start));
            anchored_end_deltas.push(abs_diff_ms(anchored.end_ms, ref_end));
            let mid_delta = abs_diff_ms(anchored_mid, ref_mid);
            anchored_mid_deltas.push(mid_delta);
            utterance_anchor_mid_deltas.push(mid_delta);
            anchored_signed_pairs.push(SignedBoundaryPair {
                start_ms: signed_diff_ms(anchored.start_ms, ref_start),
                end_ms: signed_diff_ms(anchored.end_ms, ref_end),
            });
        }

        total_hypothesis_words += hyp_norm.len();
        total_matched_words += matches.len();
        total_anchored_words += anchored_words.len();
        total_edit_distance += edit_distance;
        summaries.push(UtteranceSummary {
            key: key.clone(),
            duration,
            reference_words: ref_norm.len(),
            hypothesis_words: hyp_norm.len(),
            matched_words: matches.len(),
            word_error_rate: edit_distance as f64 / ref_norm.len().max(1) as f64,
            anchored_mean_mid_ms: mean(&utterance_anchor_mid_deltas),
            mean_decode_ms,
            text: result.text,
        });

        eprintln!(
            "utterance={} key={} decode_ms={:.2} rtfx={:.2} ref_words={} hyp_words={} matched={}",
            seq + 1,
            key,
            mean_decode_ms,
            duration / (mean_decode_ms / 1000.0),
            ref_norm.len(),
            hyp_norm.len(),
            matches.len()
        );
    }

    all_mid_deltas.sort_unstable();
    anchored_mid_deltas.sort_unstable();
    decode_times.sort_by(f64::total_cmp);
    let total_mean_decode_ms = summaries
        .iter()
        .map(|summary| summary.mean_decode_ms)
        .sum::<f64>();
    let output = EvalOutput {
        result: EvalResult {
            utterances: summaries.len(),
            failures,
            audio_seconds: total_audio_seconds,
            reference_words: total_reference_words,
            hypothesis_words: total_hypothesis_words,
            matched_words: total_matched_words,
            word_error_rate: total_edit_distance as f64 / total_reference_words.max(1) as f64,
            anchored_words: total_anchored_words,
            anchored_mean_start_ms: mean(&anchored_start_deltas),
            anchored_mean_end_ms: mean(&anchored_end_deltas),
            anchored_mean_mid_ms: mean(&anchored_mid_deltas),
            anchored_p50_mid_ms: percentile(&anchored_mid_deltas, 50.0),
            anchored_p90_mid_ms: percentile(&anchored_mid_deltas, 90.0),
            anchored_p95_mid_ms: percentile(&anchored_mid_deltas, 95.0),
            anchored_p99_mid_ms: percentile(&anchored_mid_deltas, 99.0),
            direct_calibrated: calibrated_metrics(&all_signed_pairs),
            anchored_calibrated: calibrated_metrics(&anchored_signed_pairs),
            total_mean_decode_ms,
            rtfx: total_audio_seconds / (total_mean_decode_ms / 1000.0),
            mean_start_ms: mean(&all_start_deltas),
            mean_end_ms: mean(&all_end_deltas),
            mean_mid_ms: mean(&all_mid_deltas),
            p50_mid_ms: percentile(&all_mid_deltas, 50.0),
            p90_mid_ms: percentile(&all_mid_deltas, 90.0),
            p95_mid_ms: percentile(&all_mid_deltas, 95.0),
            p99_mid_ms: percentile(&all_mid_deltas, 99.0),
            per_utterance_mean_decode_ms: mean_f64(&decode_times),
            per_utterance_p50_decode_ms: percentile_f64(&decode_times, 50.0),
            per_utterance_p95_decode_ms: percentile_f64(&decode_times, 95.0),
        },
        summary: summaries,
    };

    let json = serde_json::to_string_pretty(&output)?;
    println!("{json}");
    if let Some(path) = &args.output_json {
        std::fs::write(path, json)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }

    std::mem::forget(aligner);
    std::process::exit(0);
}

fn load_alignments(path: &Path) -> Result<BTreeMap<String, ReferenceAlignment>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    serde_json::from_reader(file).with_context(|| format!("failed to parse {}", path.display()))
}

async fn decode_audio(path: &Path) -> Result<Vec<f32>> {
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

fn align_words<'a>(
    hypothesis: &'a [TimedWord],
    reference: &'a [ReferenceWord],
) -> Vec<MatchedWord<'a>> {
    let left = hypothesis
        .iter()
        .map(|word| normalize_word(&word.word))
        .collect::<Vec<_>>();
    let right = reference
        .iter()
        .map(|word| normalize_word(&word.word))
        .collect::<Vec<_>>();
    let mut dp = vec![vec![0usize; right.len() + 1]; left.len() + 1];
    for i in (0..left.len()).rev() {
        for j in (0..right.len()).rev() {
            dp[i][j] = if left[i] == right[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }

    let mut matches = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left.len() && j < right.len() {
        if left[i] == right[j] && !left[i].is_empty() {
            matches.push(MatchedWord {
                hypothesis: &hypothesis[i],
                reference: &reference[j],
            });
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    matches
}

fn timestamp_reference_from_hypothesis(
    hypothesis: &[TimedWord],
    reference: &[ReferenceWord],
    sample_count: usize,
) -> Vec<TimedWord> {
    if reference.is_empty() {
        return Vec::new();
    }

    let matches = align_word_indices(hypothesis, reference);
    let mut output = vec![None; reference.len()];
    for (hyp_index, ref_index) in &matches {
        let hyp_word = &hypothesis[*hyp_index];
        output[*ref_index] = Some(TimedWord {
            word: reference[*ref_index].word.clone(),
            start_ms: hyp_word.start_ms,
            end_ms: hyp_word.end_ms,
        });
    }

    let duration_ms = ((sample_count as f64 / f64::from(ASR_SAMPLE_RATE)) * 1000.0)
        .round()
        .clamp(0.0, f64::from(u32::MAX)) as u32;
    let mut previous_ref: Option<usize> = None;
    for (_, next_ref) in matches
        .iter()
        .map(|(_, ref_index)| *ref_index)
        .chain(std::iter::once(reference.len()))
        .enumerate()
    {
        let gap_start = previous_ref.map_or(0, |index| index + 1);
        let gap_end = next_ref;
        if gap_start < gap_end {
            let left_ms = previous_ref
                .and_then(|index| output[index].as_ref().map(|word| word.end_ms))
                .unwrap_or(0);
            let right_ms = if next_ref < reference.len() {
                output[next_ref]
                    .as_ref()
                    .map(|word| word.start_ms)
                    .unwrap_or(duration_ms)
            } else {
                duration_ms
            };
            fill_reference_gap(
                &mut output,
                reference,
                gap_start,
                gap_end,
                left_ms,
                right_ms,
            );
        }
        if next_ref < reference.len() {
            previous_ref = Some(next_ref);
        }
    }

    output.into_iter().flatten().collect()
}

fn fill_reference_gap(
    output: &mut [Option<TimedWord>],
    reference: &[ReferenceWord],
    start_index: usize,
    end_index: usize,
    left_ms: u32,
    right_ms: u32,
) {
    let count = end_index.saturating_sub(start_index);
    if count == 0 {
        return;
    }

    let available_ms = right_ms.saturating_sub(left_ms);
    let weights = reference[start_index..end_index]
        .iter()
        .map(|word| normalize_word(&word.word).len().max(1) as u32)
        .collect::<Vec<_>>();
    let total_weight = weights.iter().sum::<u32>().max(1);
    let mut consumed = 0u32;
    for (offset, weight) in weights.iter().enumerate() {
        let word_index = start_index + offset;
        let start_ms = left_ms
            + ((u64::from(available_ms) * u64::from(consumed)) / u64::from(total_weight)) as u32;
        consumed += *weight;
        let mut end_ms = left_ms
            + ((u64::from(available_ms) * u64::from(consumed)) / u64::from(total_weight)) as u32;
        if end_ms <= start_ms {
            end_ms = start_ms.saturating_add(1);
        }
        output[word_index] = Some(TimedWord {
            word: reference[word_index].word.clone(),
            start_ms,
            end_ms,
        });
    }
}

fn align_word_indices(
    hypothesis: &[TimedWord],
    reference: &[ReferenceWord],
) -> Vec<(usize, usize)> {
    let left = hypothesis
        .iter()
        .map(|word| normalize_word(&word.word))
        .collect::<Vec<_>>();
    let right = reference
        .iter()
        .map(|word| normalize_word(&word.word))
        .collect::<Vec<_>>();
    let mut dp = vec![vec![0usize; right.len() + 1]; left.len() + 1];
    for i in (0..left.len()).rev() {
        for j in (0..right.len()).rev() {
            dp[i][j] = if left[i] == right[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }

    let mut matches = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left.len() && j < right.len() {
        if left[i] == right[j] && !left[i].is_empty() {
            matches.push((i, j));
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    matches
}

fn edit_distance(left: &[String], right: &[String]) -> usize {
    let mut prev = (0..=right.len()).collect::<Vec<_>>();
    let mut curr = vec![0usize; right.len() + 1];
    for (i, left_word) in left.iter().enumerate() {
        curr[0] = i + 1;
        for (j, right_word) in right.iter().enumerate() {
            curr[j + 1] = if left_word == right_word {
                prev[j]
            } else {
                (prev[j] + 1).min(prev[j + 1] + 1).min(curr[j] + 1)
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[right.len()]
}

fn normalize_word(word: &str) -> String {
    word.chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '\'')
        .flat_map(char::to_lowercase)
        .collect()
}

fn seconds_to_ms(seconds: f64) -> u32 {
    (seconds * 1000.0).round().clamp(0.0, f64::from(u32::MAX)) as u32
}

fn midpoint_ms(start_ms: u32, end_ms: u32) -> u32 {
    start_ms.saturating_add(end_ms).saturating_div(2)
}

fn abs_diff_ms(left: u32, right: u32) -> u32 {
    left.max(right) - left.min(right)
}

fn signed_diff_ms(left: u32, right: u32) -> i32 {
    left as i32 - right as i32
}

fn mean(values: &[u32]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(sorted_values: &[u32], percentile: f64) -> u32 {
    if sorted_values.is_empty() {
        return 0;
    }
    let index =
        ((percentile / 100.0) * (sorted_values.len().saturating_sub(1) as f64)).round() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}

fn percentile_f64(sorted_values: &[f64], percentile: f64) -> f64 {
    if sorted_values.is_empty() {
        return f64::NAN;
    }
    let index =
        ((percentile / 100.0) * (sorted_values.len().saturating_sub(1) as f64)).round() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}

fn calibrated_metrics(pairs: &[SignedBoundaryPair]) -> CalibratedMetrics {
    let start_offset_ms = -median_signed(pairs.iter().map(|pair| pair.start_ms));
    let end_offset_ms = -median_signed(pairs.iter().map(|pair| pair.end_ms));
    let mut start_deltas = Vec::with_capacity(pairs.len());
    let mut end_deltas = Vec::with_capacity(pairs.len());
    let mut mid_deltas = Vec::with_capacity(pairs.len());

    for pair in pairs {
        let start = pair.start_ms + start_offset_ms;
        let end = pair.end_ms + end_offset_ms;
        start_deltas.push(start.unsigned_abs());
        end_deltas.push(end.unsigned_abs());
        mid_deltas.push(((start + end) / 2).unsigned_abs());
    }
    mid_deltas.sort_unstable();

    CalibratedMetrics {
        start_offset_ms,
        end_offset_ms,
        mean_start_ms: mean(&start_deltas),
        mean_end_ms: mean(&end_deltas),
        mean_mid_ms: mean(&mid_deltas),
        p50_mid_ms: percentile(&mid_deltas, 50.0),
        p90_mid_ms: percentile(&mid_deltas, 90.0),
        p95_mid_ms: percentile(&mid_deltas, 95.0),
        p99_mid_ms: percentile(&mid_deltas, 99.0),
    }
}

fn median_signed(values: impl Iterator<Item = i32>) -> i32 {
    let mut values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}
