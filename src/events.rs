use bytes::Bytes;
use serde::Serialize;

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TranscriptEvent {
    Started {
        stream_id: u64,
        chunk_ms: u64,
        overlap_ms: u64,
    },
    Segment {
        index: u64,
        start_ms: u64,
        end_ms: u64,
        text: String,
        final_segment: bool,
    },
    Error {
        message: String,
    },
    Done {
        segment_count: u64,
    },
}

pub fn encode_event(event: &TranscriptEvent) -> anyhow::Result<Bytes> {
    let mut line = serde_json::to_vec(event)?;
    line.push(b'\n');
    Ok(Bytes::from(line))
}
