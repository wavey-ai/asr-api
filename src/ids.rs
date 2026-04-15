use crate::protocol::{INTERNAL_REQUEST_ID_HEADER, PUBLIC_REQUEST_ID_HEADER};
use gen_id::{ConfigPreset::ShortEpochMaxNodes, IdGenerator, DEFAULT_EPOCH};
use http::HeaderMap;
use std::sync::OnceLock;

fn generator() -> &'static IdGenerator {
    static GENERATOR: OnceLock<IdGenerator> = OnceLock::new();
    GENERATOR.get_or_init(|| IdGenerator::new(ShortEpochMaxNodes, DEFAULT_EPOCH))
}

pub fn next_request_id() -> i64 {
    let raw = generator().next_id(1) & (i64::MAX as u64);
    i64::try_from(raw).unwrap_or(i64::MAX)
}

pub fn request_id_from_headers(headers: &HeaderMap) -> Option<i64> {
    [INTERNAL_REQUEST_ID_HEADER, PUBLIC_REQUEST_ID_HEADER]
        .into_iter()
        .find_map(|name| {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.trim().parse::<i64>().ok())
                .filter(|request_id| *request_id > 0)
        })
}

pub fn ensure_request_id(headers: &HeaderMap) -> i64 {
    request_id_from_headers(headers).unwrap_or_else(next_request_id)
}
