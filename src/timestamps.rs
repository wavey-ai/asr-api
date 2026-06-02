use crate::chunking::TimedWord;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TokenTextSpan {
    pub token_index: usize,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy)]
struct TextSpan {
    start: usize,
    end: usize,
}

pub(crate) fn duration_ms_for_samples(sample_count: usize, sample_rate: u32) -> u32 {
    if sample_rate == 0 {
        return 0;
    }
    ((sample_count as f64 / f64::from(sample_rate)) * 1000.0)
        .round()
        .clamp(0.0, f64::from(u32::MAX)) as u32
}

pub(crate) fn estimate_word_timestamps_from_tokens(
    text: &str,
    token_spans: &[TokenTextSpan],
    token_count: usize,
    duration_ms: u32,
) -> Vec<TimedWord> {
    let text = text.trim();
    let word_spans = whitespace_word_spans(text);
    if text.is_empty() || word_spans.is_empty() || duration_ms == 0 {
        return Vec::new();
    }

    let effective_token_count = token_count
        .max(token_spans.len())
        .max(word_spans.len())
        .max(1);
    let text_len = text.len().max(1);
    let mut words = Vec::with_capacity(word_spans.len());
    let mut last_end_ms = 0u32;

    for span in word_spans {
        let (start_token, end_token) =
            token_bounds_for_word(span, token_spans, effective_token_count, text_len);
        let mut start_ms = token_to_ms(start_token, effective_token_count, duration_ms);
        let mut end_ms = token_to_ms(end_token, effective_token_count, duration_ms);

        start_ms = start_ms.max(last_end_ms);
        if end_ms <= start_ms {
            end_ms = start_ms.saturating_add(1).min(duration_ms.max(start_ms));
        }
        last_end_ms = end_ms;

        words.push(TimedWord {
            word: text[span.start..span.end].to_string(),
            start_ms,
            end_ms,
        });
    }

    words
}

#[cfg(any(feature = "cohere-mlx", test))]
pub(crate) fn estimate_word_timestamps_from_token_count(
    text: &str,
    token_count: usize,
    duration_ms: u32,
) -> Vec<TimedWord> {
    estimate_word_timestamps_from_tokens(text, &[], token_count, duration_ms)
}

fn token_bounds_for_word(
    word: TextSpan,
    token_spans: &[TokenTextSpan],
    token_count: usize,
    text_len: usize,
) -> (usize, usize) {
    let mut first = None;
    let mut last = None;

    for token in token_spans {
        if token.end > word.start && token.start < word.end {
            first.get_or_insert(token.token_index);
            last = Some(token.token_index + 1);
        }
    }

    if let (Some(first), Some(last)) = (first, last) {
        return (first.min(token_count), last.max(first + 1).min(token_count));
    }

    let start = (word.start * token_count) / text_len;
    let mut end = word.end.saturating_mul(token_count).div_ceil(text_len);
    end = end.max(start + 1).min(token_count);
    (start.min(token_count.saturating_sub(1)), end)
}

fn token_to_ms(token_index: usize, token_count: usize, duration_ms: u32) -> u32 {
    if token_count == 0 {
        return 0;
    }
    ((token_index as f64 / token_count as f64) * f64::from(duration_ms))
        .round()
        .clamp(0.0, f64::from(duration_ms)) as u32
}

fn whitespace_word_spans(text: &str) -> Vec<TextSpan> {
    let mut spans = Vec::new();
    let mut start = None;

    for (index, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(word_start) = start.take() {
                spans.push(TextSpan {
                    start: word_start,
                    end: index,
                });
            }
        } else if start.is_none() {
            start = Some(index);
        }
    }

    if let Some(word_start) = start {
        spans.push(TextSpan {
            start: word_start,
            end: text.len(),
        });
    }

    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_follow_decoded_token_spans() {
        let words = estimate_word_timestamps_from_tokens(
            "hello world",
            &[
                TokenTextSpan {
                    token_index: 0,
                    start: 0,
                    end: 2,
                },
                TokenTextSpan {
                    token_index: 1,
                    start: 2,
                    end: 5,
                },
                TokenTextSpan {
                    token_index: 2,
                    start: 6,
                    end: 11,
                },
            ],
            3,
            3000,
        );

        assert_eq!(
            words,
            vec![
                TimedWord {
                    word: "hello".into(),
                    start_ms: 0,
                    end_ms: 2000,
                },
                TimedWord {
                    word: "world".into(),
                    start_ms: 2000,
                    end_ms: 3000,
                },
            ]
        );
    }

    #[test]
    fn token_count_fallback_is_monotonic() {
        let words = estimate_word_timestamps_from_token_count(
            "ask not what your country can do for you",
            12,
            4000,
        );

        assert_eq!(words.len(), 9);
        assert_eq!(words[0].word, "ask");
        assert_eq!(words.last().unwrap().word, "you");
        for pair in words.windows(2) {
            assert!(pair[0].end_ms <= pair[1].start_ms);
        }
        assert_eq!(words.last().unwrap().end_ms, 4000);
    }

    #[test]
    fn duration_uses_sample_rate() {
        assert_eq!(duration_ms_for_samples(16_000, 16_000), 1000);
        assert_eq!(duration_ms_for_samples(8_000, 16_000), 500);
    }
}
