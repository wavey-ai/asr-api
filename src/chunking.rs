use crate::config::PARAKEET_SAMPLE_RATE;

const SAMPLE_RATE_F64: f64 = PARAKEET_SAMPLE_RATE as f64;
const DEDUPE_EPSILON_MS: u64 = 25;

#[derive(Debug, Clone, PartialEq)]
pub struct TimedSegment {
    pub text: String,
    pub start_secs: f64,
    pub end_secs: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmittedSegment {
    pub index: u64,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
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

    pub fn chunk_samples(&self) -> usize {
        self.chunk_samples
    }

    pub fn stride_samples(&self) -> usize {
        self.stride_samples
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
        Some(AudioWindow {
            samples: std::mem::take(&mut self.pending),
            start_sample: self.start_sample,
            is_final: true,
        })
    }
}

#[derive(Debug, Default)]
pub struct SegmentCommitter {
    next_index: u64,
    last_emitted_end_ms: Option<u64>,
}

impl SegmentCommitter {
    pub fn commit(
        &mut self,
        window_start_sample: usize,
        stable_samples: usize,
        is_final: bool,
        segments: &[TimedSegment],
    ) -> Vec<EmittedSegment> {
        let stable_limit_secs = stable_samples as f64 / SAMPLE_RATE_F64;
        let window_start_ms = sample_to_ms(window_start_sample);
        let mut emitted = Vec::new();

        for segment in segments {
            if segment.text.trim().is_empty() {
                continue;
            }
            if !is_final && segment.end_secs > stable_limit_secs {
                continue;
            }

            let start_ms = window_start_ms + seconds_to_ms(segment.start_secs);
            let end_ms = window_start_ms + seconds_to_ms(segment.end_secs);

            if let Some(last_end_ms) = self.last_emitted_end_ms {
                if end_ms <= last_end_ms.saturating_add(DEDUPE_EPSILON_MS) {
                    continue;
                }
            }

            emitted.push(EmittedSegment {
                index: self.next_index,
                start_ms,
                end_ms,
                text: segment.text.clone(),
            });
            self.next_index += 1;
            self.last_emitted_end_ms = Some(end_ms);
        }

        emitted
    }

    pub fn emitted_count(&self) -> u64 {
        self.next_index
    }
}

fn sample_to_ms(sample: usize) -> u64 {
    ((sample as f64 / SAMPLE_RATE_F64) * 1000.0).round() as u64
}

fn seconds_to_ms(seconds: f64) -> u64 {
    (seconds * 1000.0).round() as u64
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
        let mut committer = SegmentCommitter::default();

        let first = committer.commit(
            0,
            8 * PARAKEET_SAMPLE_RATE as usize,
            false,
            &[
                TimedSegment {
                    text: "alpha".into(),
                    start_secs: 0.2,
                    end_secs: 1.0,
                },
                TimedSegment {
                    text: "beta".into(),
                    start_secs: 7.5,
                    end_secs: 8.4,
                },
            ],
        );
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].text, "alpha");

        let second = committer.commit(
            8 * PARAKEET_SAMPLE_RATE as usize,
            8 * PARAKEET_SAMPLE_RATE as usize,
            true,
            &[
                TimedSegment {
                    text: "beta".into(),
                    start_secs: 0.0,
                    end_secs: 0.4,
                },
                TimedSegment {
                    text: "gamma".into(),
                    start_secs: 1.0,
                    end_secs: 1.8,
                },
            ],
        );

        assert_eq!(second.len(), 2);
        assert_eq!(second[0].text, "beta");
        assert_eq!(second[1].text, "gamma");
    }
}
