use crate::config::ASR_SAMPLE_RATE;

#[derive(Debug)]
pub struct Linear16PcmStream {
    channels: usize,
    pending_bytes: Vec<u8>,
    resampler: Option<LinearResampler>,
}

impl Linear16PcmStream {
    pub fn new(input_sample_rate: u32, channels: u8) -> Result<Self, String> {
        if input_sample_rate == 0 {
            return Err("sample_rate must be greater than 0".into());
        }
        if channels == 0 {
            return Err("channels must be greater than 0".into());
        }

        Ok(Self {
            channels: channels as usize,
            pending_bytes: Vec::new(),
            resampler: (input_sample_rate != ASR_SAMPLE_RATE)
                .then(|| LinearResampler::new(input_sample_rate, ASR_SAMPLE_RATE)),
        })
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<f32>, String> {
        if bytes.is_empty() {
            return Ok(Vec::new());
        }

        self.pending_bytes.extend_from_slice(bytes);
        let bytes_per_frame = self.channels * std::mem::size_of::<i16>();
        let complete_len = self.pending_bytes.len() / bytes_per_frame * bytes_per_frame;
        if complete_len == 0 {
            return Ok(Vec::new());
        }

        let drained = self.pending_bytes.drain(..complete_len).collect::<Vec<_>>();
        let mono = decode_linear16_mono(&drained, self.channels);
        if let Some(resampler) = &mut self.resampler {
            Ok(resampler.push(&mono))
        } else {
            Ok(mono)
        }
    }

    pub fn finish(&mut self) -> Result<Vec<f32>, String> {
        if !self.pending_bytes.is_empty() {
            return Err(format!(
                "raw linear16 body ended with {} trailing bytes",
                self.pending_bytes.len()
            ));
        }

        if let Some(resampler) = &mut self.resampler {
            Ok(resampler.finish())
        } else {
            Ok(Vec::new())
        }
    }
}

pub fn rms_level(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum_squares = samples
        .iter()
        .map(|sample| {
            let sample = *sample;
            sample * sample
        })
        .sum::<f32>();
    (sum_squares / samples.len() as f32).sqrt()
}

fn decode_linear16_mono(bytes: &[u8], channels: usize) -> Vec<f32> {
    let bytes_per_frame = channels * std::mem::size_of::<i16>();
    let frames = bytes.len() / bytes_per_frame;
    let mut samples = Vec::with_capacity(frames);

    for frame in bytes.chunks_exact(bytes_per_frame) {
        let mut sum = 0.0f32;
        for channel_bytes in frame.chunks_exact(std::mem::size_of::<i16>()) {
            let sample = i16::from_le_bytes([channel_bytes[0], channel_bytes[1]]);
            sum += sample as f32 / 32768.0;
        }
        samples.push(sum / channels as f32);
    }

    samples
}

#[derive(Debug)]
struct LinearResampler {
    ratio: f64,
    next_source_pos: f64,
    source: Vec<f32>,
}

impl LinearResampler {
    fn new(input_rate: u32, output_rate: u32) -> Self {
        Self {
            ratio: input_rate as f64 / output_rate as f64,
            next_source_pos: 0.0,
            source: Vec::new(),
        }
    }

    fn push(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        self.source.extend_from_slice(input);
        self.drain_ready(false)
    }

    fn finish(&mut self) -> Vec<f32> {
        self.drain_ready(true)
    }

    fn drain_ready(&mut self, include_tail: bool) -> Vec<f32> {
        let mut output = Vec::new();
        let limit = if include_tail {
            self.source.len() as f64
        } else {
            self.source.len().saturating_sub(1) as f64
        };

        while self.next_source_pos < limit {
            let index = self.next_source_pos.floor() as usize;
            let frac = (self.next_source_pos - index as f64) as f32;
            let current = self.source[index];
            let next = self.source.get(index + 1).copied().unwrap_or(current);
            output.push(current + (next - current) * frac);
            self.next_source_pos += self.ratio;
        }

        let keep_from = self.next_source_pos.floor().max(1.0) as usize - 1;
        if keep_from > 0 {
            self.source.drain(..keep_from);
            self.next_source_pos -= keep_from as f64;
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear16_stream_passes_through_16k_mono() {
        let mut stream = Linear16PcmStream::new(16_000, 1).unwrap();
        let input = [0i16, 16384, -16384]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>();
        let output = stream.push(&input).unwrap();
        assert_eq!(output.len(), 3);
        assert!(stream.finish().unwrap().is_empty());
    }

    #[test]
    fn linear16_stream_downmixes_stereo() {
        let mut stream = Linear16PcmStream::new(16_000, 2).unwrap();
        let input = [32767i16, -32768, 16384, 16384]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>();
        let output = stream.push(&input).unwrap();
        assert_eq!(output.len(), 2);
        assert!(output[0].abs() < 0.01);
        assert!(output[1] > 0.45);
    }

    #[test]
    fn linear16_stream_resamples_8k_to_16k() {
        let mut stream = Linear16PcmStream::new(8_000, 1).unwrap();
        let input = [0i16, 32767, 0]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>();
        let mut output = stream.push(&input).unwrap();
        output.extend(stream.finish().unwrap());
        assert!(output.len() >= 5);
    }
}
