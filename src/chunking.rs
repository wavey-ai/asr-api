use crate::config::ASR_SAMPLE_RATE;

const SAMPLE_RATE_F64: f64 = ASR_SAMPLE_RATE as f64;
const DEDUPE_EPSILON_MS: u64 = 25;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimedWord {
    pub word: String,
    pub start_ms: u32,
    pub end_ms: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedWord {
    pub index: u64,
    pub start_ms: u64,
    pub end_ms: u64,
    pub word: String,
}

#[derive(Debug, Clone)]
pub struct AudioWindow {
    pub samples: Vec<f32>,
    pub start_sample: usize,
    pub is_final: bool,
}

#[derive(Debug)]
pub struct AudioChunker {
    chunk_samples: usize,
    stride_samples: usize,
    min_final_samples: usize,
    start_sample: usize,
    pending: Vec<f32>,
}

impl AudioChunker {
    pub fn new(chunk_samples: usize, overlap_samples: usize, min_final_samples: usize) -> Self {
        let stride_samples = chunk_samples.saturating_sub(overlap_samples).max(1);
        Self {
            chunk_samples,
            stride_samples,
            min_final_samples,
            start_sample: 0,
            pending: Vec::new(),
        }
    }

    pub fn stride_samples(&self) -> usize {
        self.stride_samples
    }

    pub fn pending_samples(&self) -> &[f32] {
        &self.pending
    }

    pub fn pending_start_sample(&self) -> usize {
        self.start_sample
    }

    pub fn push(&mut self, samples: &[f32]) {
        self.pending.extend_from_slice(samples);
    }

    pub fn take_ready_windows(&mut self) -> Vec<AudioWindow> {
        let mut ready = Vec::new();
        while self.pending.len() >= self.chunk_samples {
            ready.push(AudioWindow {
                samples: self.pending[..self.chunk_samples].to_vec(),
                start_sample: self.start_sample,
                is_final: false,
            });
            self.pending.drain(..self.stride_samples);
            self.start_sample += self.stride_samples;
        }
        ready
    }

    pub fn take_final_window(&mut self) -> Option<AudioWindow> {
        if self.pending.len() < self.min_final_samples {
            self.pending.clear();
            return None;
        }
        let samples = std::mem::take(&mut self.pending);
        let start_sample = self.start_sample;
        self.start_sample += samples.len();
        Some(AudioWindow {
            samples,
            start_sample,
            is_final: true,
        })
    }
}

#[derive(Debug, Default)]
pub struct WordCommitter {
    next_index: u64,
    last_emitted_end_ms: Option<u64>,
}

impl WordCommitter {
    pub fn commit(
        &mut self,
        window_start_sample: usize,
        stable_samples: usize,
        is_final: bool,
        words: &[TimedWord],
    ) -> Vec<CommittedWord> {
        let stable_limit_ms = sample_to_ms(stable_samples);
        let window_start_ms = sample_to_ms(window_start_sample);
        let mut emitted = Vec::new();

        for word in words {
            if word.word.trim().is_empty() {
                continue;
            }
            if !is_final && u64::from(word.end_ms) > stable_limit_ms {
                continue;
            }

            let start_ms = window_start_ms + u64::from(word.start_ms);
            let end_ms = window_start_ms + u64::from(word.end_ms);

            if let Some(last_end_ms) = self.last_emitted_end_ms {
                if end_ms <= last_end_ms.saturating_add(DEDUPE_EPSILON_MS) {
                    continue;
                }
            }

            emitted.push(CommittedWord {
                index: self.next_index,
                start_ms,
                end_ms,
                word: word.word.clone(),
            });
            self.next_index += 1;
            self.last_emitted_end_ms = Some(end_ms);
        }

        emitted
    }
}

fn sample_to_ms(sample: usize) -> u64 {
    ((sample as f64 / SAMPLE_RATE_F64) * 1000.0).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunker_emits_stride_windows_and_final_tail() {
        let mut chunker = AudioChunker::new(10, 2, 1);
        chunker.push(&vec![0.0; 18]);
        let windows = chunker.take_ready_windows();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].start_sample, 0);
        assert_eq!(windows[1].start_sample, 8);
        assert_eq!(windows[0].samples.len(), 10);
        assert_eq!(windows[1].samples.len(), 10);

        let tail = chunker.take_final_window().unwrap();
        assert!(tail.is_final);
        assert_eq!(tail.start_sample, 16);
        assert_eq!(tail.samples.len(), 2);
    }

    #[test]
    fn committer_skips_unstable_overlap_and_dedupes_final_tail() {
        let mut committer = WordCommitter::default();

        let first = committer.commit(
            0,
            8 * ASR_SAMPLE_RATE as usize,
            false,
            &[
                TimedWord {
                    word: "alpha".into(),
                    start_ms: 200,
                    end_ms: 1_000,
                },
                TimedWord {
                    word: "beta".into(),
                    start_ms: 7_500,
                    end_ms: 8_400,
                },
            ],
        );
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].word, "alpha");

        let second = committer.commit(
            8 * ASR_SAMPLE_RATE as usize,
            8 * ASR_SAMPLE_RATE as usize,
            true,
            &[
                TimedWord {
                    word: "beta".into(),
                    start_ms: 0,
                    end_ms: 400,
                },
                TimedWord {
                    word: "gamma".into(),
                    start_ms: 1_000,
                    end_ms: 1_800,
                },
            ],
        );

        assert_eq!(second.len(), 2);
        assert_eq!(second[0].word, "beta");
        assert_eq!(second[1].word, "gamma");
    }
}
