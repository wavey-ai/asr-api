use crate::config::ASR_SAMPLE_RATE;
use anyhow::{anyhow, Result};
use bytes::Bytes;
use serde::{Deserialize, Serialize};

pub const PROCESSING_ENCODING_F32LE: &str = "f32le";
pub const DECODE_STAGE: &str = "decoded";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessingHead {
    pub encoding: String,
    pub sample_rate_hz: u32,
    pub channels: u8,
}

impl ProcessingHead {
    pub fn pcm_mono_f32() -> Self {
        Self {
            encoding: PROCESSING_ENCODING_F32LE.to_string(),
            sample_rate_hz: ASR_SAMPLE_RATE,
            channels: 1,
        }
    }

    pub fn validate_for_asr(&self) -> Result<()> {
        anyhow::ensure!(
            self.encoding == PROCESSING_ENCODING_F32LE,
            "unsupported processing encoding: {}",
            self.encoding
        );
        anyhow::ensure!(
            self.sample_rate_hz == ASR_SAMPLE_RATE,
            "unsupported processing sample rate: {}",
            self.sample_rate_hz
        );
        anyhow::ensure!(
            self.channels == 1,
            "unsupported processing channel count: {}",
            self.channels
        );
        Ok(())
    }
}

pub fn encode_processing_head(head: &ProcessingHead) -> Result<Bytes> {
    let json =
        serde_json::to_vec(head).map_err(|error| anyhow!("encode processing head: {error}"))?;
    Ok(Bytes::from(json))
}

pub fn decode_processing_head(bytes: &[u8]) -> Result<ProcessingHead> {
    serde_json::from_slice(bytes).map_err(|error| anyhow!("decode processing head: {error}"))
}

pub fn encode_samples_f32le(samples: &[f32]) -> Bytes {
    let mut encoded = Vec::with_capacity(samples.len() * std::mem::size_of::<f32>());
    for sample in samples {
        encoded.extend_from_slice(&sample.to_le_bytes());
    }
    Bytes::from(encoded)
}

pub fn decode_samples_f32le(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    anyhow::ensure!(
        bytes.len() % std::mem::size_of::<f32>() == 0,
        "invalid f32le payload length: {}",
        bytes.len()
    );
    let mut samples = Vec::with_capacity(bytes.len() / std::mem::size_of::<f32>());
    for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
        let raw: [u8; 4] = chunk
            .try_into()
            .map_err(|_| anyhow!("invalid f32le sample payload"))?;
        samples.push(f32::from_le_bytes(raw));
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_processing_head() {
        let head = ProcessingHead::pcm_mono_f32();
        let encoded = encode_processing_head(&head).unwrap();
        let decoded = decode_processing_head(&encoded).unwrap();
        assert_eq!(decoded, head);
    }

    #[test]
    fn round_trips_f32le_samples() {
        let samples = vec![0.0, 1.25, -3.5];
        let encoded = encode_samples_f32le(&samples);
        let decoded = decode_samples_f32le(&encoded).unwrap();
        assert_eq!(decoded, samples);
    }
}
