use crate::chunking::CommittedWord;
use crate::config::{AppConfig, DEFAULT_LANGUAGE};
use chrono::{SecondsFormat, Utc};
use http::Request;
use serde::Serialize;
use std::collections::BTreeMap;
use url::form_urlencoded;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioEncoding {
    Linear16,
}

#[derive(Debug, Clone)]
pub struct ListenOptions {
    pub utterances: bool,
    pub paragraphs: bool,
    pub timestamps: bool,
    pub words: bool,
    pub interim_results: bool,
    pub endpointing_ms: Option<u64>,
    pub utterance_split_secs: f64,
    pub language: String,
    pub model: String,
    pub encoding: Option<AudioEncoding>,
    pub sample_rate_hz: Option<u32>,
    pub channels: u8,
    pub diarize: bool,
    pub punctuate: bool,
    pub smart_format: bool,
    pub numerals: bool,
}

impl ListenOptions {
    pub fn from_request(req: &Request<()>, config: &AppConfig) -> Self {
        let header_has_timestamps = req.headers().get("timestamps").is_some();
        let header_has_words = req.headers().get("words").is_some();
        let mut options = Self {
            utterances: false,
            paragraphs: false,
            timestamps: false,
            words: false,
            interim_results: false,
            endpointing_ms: None,
            utterance_split_secs: config.utt_split_seconds,
            language: DEFAULT_LANGUAGE.to_string(),
            model: config.default_model_name().to_string(),
            encoding: None,
            sample_rate_hz: None,
            channels: 1,
            diarize: false,
            punctuate: true,
            smart_format: false,
            numerals: false,
        };

        if let Some(value) = get_request_value(req, "timestamps", "timestamps") {
            options.timestamps = parse_bool(&value).unwrap_or(false);
        } else if let Some(value) = get_request_value(req, "words", "words") {
            options.words = parse_bool(&value).unwrap_or(false);
        }

        if let Some(query) = req.uri().query() {
            for (key, value) in form_urlencoded::parse(query.as_bytes()) {
                match key.as_ref() {
                    "utterances" => {
                        if let Some(parsed) = parse_bool(&value) {
                            options.utterances = parsed;
                        }
                    }
                    "paragraphs" => {
                        if let Some(parsed) = parse_bool(&value) {
                            options.paragraphs = parsed;
                        }
                    }
                    "timestamps" => {
                        if !header_has_timestamps {
                            if let Some(parsed) = parse_bool(&value) {
                                options.timestamps = parsed;
                            }
                        }
                    }
                    "words" => {
                        if !header_has_words {
                            if let Some(parsed) = parse_bool(&value) {
                                options.words = parsed;
                            }
                        }
                    }
                    "interim_results" => {
                        if let Some(parsed) = parse_bool(&value) {
                            options.interim_results = parsed;
                        }
                    }
                    "endpointing" => {
                        options.endpointing_ms = parse_u64(&value);
                    }
                    "utt_split" => {
                        if let Ok(parsed) = value.parse::<f64>() {
                            if parsed >= 0.0 {
                                options.utterance_split_secs = parsed;
                            }
                        }
                    }
                    "language" => {
                        if !value.is_empty() {
                            options.language = value.into_owned();
                        }
                    }
                    "model" => {
                        if !value.is_empty() {
                            options.model = value.into_owned();
                        }
                    }
                    "encoding" => {
                        options.encoding = parse_encoding(&value);
                    }
                    "sample_rate" => {
                        options.sample_rate_hz = parse_sample_rate(&value);
                    }
                    "channels" => {
                        if let Some(parsed) = parse_u8(&value) {
                            options.channels = parsed.max(1);
                        }
                    }
                    "diarize" => {
                        if let Some(parsed) = parse_bool(&value) {
                            options.diarize = parsed;
                        }
                    }
                    "punctuate" => {
                        if let Some(parsed) = parse_bool(&value) {
                            options.punctuate = parsed;
                        }
                    }
                    "smart_format" => {
                        if let Some(parsed) = parse_bool(&value) {
                            options.smart_format = parsed;
                        }
                    }
                    "numerals" => {
                        if let Some(parsed) = parse_bool(&value) {
                            options.numerals = parsed;
                        }
                    }
                    _ => {}
                }
            }
        }

        if options.timestamps {
            options.words = true;
        }

        options
    }

    pub fn wants_word_timestamps(&self) -> bool {
        self.timestamps || self.words || self.utterances || self.paragraphs
    }

    pub fn raw_linear16(&self) -> bool {
        self.encoding == Some(AudioEncoding::Linear16)
    }
}

#[derive(Debug, Serialize)]
pub struct ListenResponse {
    pub metadata: Metadata,
    pub results: Results,
}

#[derive(Debug, Serialize)]
pub struct Metadata {
    pub transaction_key: String,
    pub request_id: String,
    pub sha256: String,
    pub created: String,
    pub duration: f64,
    pub channels: u32,
    pub models: Vec<String>,
    pub model_info: BTreeMap<String, ModelInfo>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub arch: String,
}

#[derive(Debug, Serialize)]
pub struct Results {
    pub channels: Vec<ChannelResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utterances: Option<Vec<Utterance>>,
}

#[derive(Debug, Serialize)]
pub struct ChannelResult {
    pub alternatives: Vec<Alternative>,
    pub detected_language: String,
}

#[derive(Debug, Serialize)]
pub struct Alternative {
    pub transcript: String,
    pub confidence: f64,
    pub words: Vec<Word>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paragraphs: Option<ParagraphResults>,
}

#[derive(Debug, Serialize, Clone)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
}

#[derive(Debug, Serialize)]
pub struct ParagraphResults {
    pub transcript: String,
    pub paragraphs: Vec<Paragraph>,
}

#[derive(Debug, Serialize)]
pub struct Paragraph {
    pub sentences: Vec<Sentence>,
    pub speaker: u8,
    pub num_words: usize,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Serialize)]
pub struct Sentence {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Serialize)]
pub struct Utterance {
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
    pub channel: u8,
    pub transcript: String,
    pub words: Vec<UtteranceWord>,
    pub speaker: u8,
    pub id: String,
}

#[derive(Debug, Serialize)]
pub struct UtteranceWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
    pub speaker: u8,
    pub punctuated_word: String,
}

pub fn default_model_info(model: &str) -> (String, BTreeMap<String, ModelInfo>) {
    let model_id = model.to_string();
    let mut model_info = BTreeMap::new();
    model_info.insert(
        model_id.clone(),
        ModelInfo {
            name: model.to_string(),
            version: "local".into(),
            arch: "cohere-transcribe-seq2seq".into(),
        },
    );
    (model_id, model_info)
}

pub(crate) fn words_from_committed(committed_words: &[CommittedWord]) -> Vec<Word> {
    committed_words
        .iter()
        .map(|word| Word {
            word: word.word.clone(),
            start: word.start_ms as f64 / 1000.0,
            end: word.end_ms as f64 / 1000.0,
            confidence: 0.0,
        })
        .collect()
}

pub fn build_response(
    request_id: String,
    sha256: String,
    duration_secs: f64,
    committed_words: &[CommittedWord],
    fallback_transcript: &str,
    options: &ListenOptions,
) -> ListenResponse {
    let transcript = transcript_from_words(committed_words, fallback_transcript);
    let words = words_from_committed(committed_words);

    let utterances = if options.utterances || options.paragraphs {
        let utterances = build_utterances(committed_words, &transcript, duration_secs, options);
        Some(utterances)
    } else {
        None
    };

    let paragraphs = if options.paragraphs {
        Some(build_paragraphs(
            utterances.as_deref().unwrap_or(&[]),
            &transcript,
            duration_secs,
        ))
    } else {
        None
    };

    let (model_id, model_info) = default_model_info(&options.model);

    ListenResponse {
        metadata: Metadata {
            transaction_key: request_id.clone(),
            request_id,
            sha256,
            created: Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
            duration: duration_secs,
            channels: 1,
            models: vec![model_id],
            model_info,
        },
        results: Results {
            channels: vec![ChannelResult {
                alternatives: vec![Alternative {
                    transcript: transcript.clone(),
                    confidence: 0.0,
                    words,
                    paragraphs,
                }],
                detected_language: options.language.clone(),
            }],
            utterances: utterances.filter(|utterances| !utterances.is_empty()),
        },
    }
}

fn build_utterances(
    committed_words: &[CommittedWord],
    transcript: &str,
    duration_secs: f64,
    options: &ListenOptions,
) -> Vec<Utterance> {
    if committed_words.is_empty() {
        if transcript.is_empty() {
            return Vec::new();
        }
        return vec![Utterance {
            start: 0.0,
            end: duration_secs,
            confidence: 0.0,
            channel: 1,
            transcript: transcript.to_string(),
            words: Vec::new(),
            speaker: 1,
            id: "utt-0001".into(),
        }];
    }

    let mut utterances = Vec::new();
    let gap_threshold_ms = (options.utterance_split_secs.max(0.0) * 1000.0).round() as u64;
    let mut start_index = 0usize;

    for index in 0..committed_words.len() {
        let is_last = index + 1 == committed_words.len();
        let boundary = if is_last {
            true
        } else {
            let current = &committed_words[index];
            let next = &committed_words[index + 1];
            let gap = next.start_ms.saturating_sub(current.end_ms);
            gap >= gap_threshold_ms || ends_sentence(&current.word)
        };

        if !boundary {
            continue;
        }

        let slice = &committed_words[start_index..=index];
        let utterance_words = slice
            .iter()
            .map(|word| UtteranceWord {
                word: word.word.clone(),
                start: word.start_ms as f64 / 1000.0,
                end: word.end_ms as f64 / 1000.0,
                confidence: 0.0,
                speaker: 1,
                punctuated_word: word.word.clone(),
            })
            .collect::<Vec<_>>();

        utterances.push(Utterance {
            start: slice.first().map(seconds_from_word_start).unwrap_or(0.0),
            end: slice
                .last()
                .map(seconds_from_word_end)
                .unwrap_or(duration_secs),
            confidence: 0.0,
            channel: 1,
            transcript: join_words(slice.iter().map(|word| word.word.as_str())),
            words: utterance_words,
            speaker: 1,
            id: format!("utt-{:04}", utterances.len() + 1),
        });
        start_index = index + 1;
    }

    utterances
}

fn build_paragraphs(
    utterances: &[Utterance],
    transcript: &str,
    duration_secs: f64,
) -> ParagraphResults {
    if utterances.is_empty() {
        return ParagraphResults {
            transcript: transcript.to_string(),
            paragraphs: if transcript.is_empty() {
                Vec::new()
            } else {
                vec![Paragraph {
                    sentences: vec![Sentence {
                        text: transcript.to_string(),
                        start: 0.0,
                        end: duration_secs,
                    }],
                    speaker: 1,
                    num_words: transcript.split_whitespace().count(),
                    start: 0.0,
                    end: duration_secs,
                }]
            },
        };
    }

    ParagraphResults {
        transcript: transcript.to_string(),
        paragraphs: vec![Paragraph {
            sentences: utterances
                .iter()
                .map(|utterance| Sentence {
                    text: utterance.transcript.clone(),
                    start: utterance.start,
                    end: utterance.end,
                })
                .collect(),
            speaker: 1,
            num_words: utterances
                .iter()
                .map(|utterance| utterance.words.len())
                .sum(),
            start: utterances
                .first()
                .map(|utterance| utterance.start)
                .unwrap_or(0.0),
            end: utterances
                .last()
                .map(|utterance| utterance.end)
                .unwrap_or(duration_secs),
        }],
    }
}

pub(crate) fn transcript_from_words(committed_words: &[CommittedWord], fallback: &str) -> String {
    if committed_words.is_empty() {
        return fallback.trim().to_string();
    }
    join_words(committed_words.iter().map(|word| word.word.as_str()))
}

pub(crate) fn join_words<'a>(words: impl Iterator<Item = &'a str>) -> String {
    let mut transcript = String::new();
    for word in words {
        append_word(&mut transcript, word);
    }
    transcript
}

pub(crate) fn append_word(transcript: &mut String, word: &str) {
    if transcript.is_empty() {
        transcript.push_str(word);
        return;
    }

    if attaches_to_previous(word) {
        transcript.push_str(word);
    } else {
        transcript.push(' ');
        transcript.push_str(word);
    }
}

pub(crate) fn attaches_to_previous(word: &str) -> bool {
    matches!(
        word,
        "." | "," | "!" | "?" | ";" | ":" | "%" | ")" | "]" | "}" | "'"
    ) || word.starts_with('\'')
        || word.starts_with('’')
}

pub(crate) fn ends_sentence(word: &str) -> bool {
    word.ends_with('.') || word.ends_with('!') || word.ends_with('?')
}

fn seconds_from_word_start(word: &CommittedWord) -> f64 {
    word.start_ms as f64 / 1000.0
}

fn seconds_from_word_end(word: &CommittedWord) -> f64 {
    word.end_ms as f64 / 1000.0
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_u64(value: &str) -> Option<u64> {
    value.trim().parse::<u64>().ok()
}

fn parse_u8(value: &str) -> Option<u8> {
    value.trim().parse::<u8>().ok()
}

fn parse_sample_rate(value: &str) -> Option<u32> {
    let sample_rate = value.trim().parse::<u32>().ok()?;
    (sample_rate > 0).then_some(sample_rate)
}

fn parse_encoding(value: &str) -> Option<AudioEncoding> {
    match value.trim().to_ascii_lowercase().as_str() {
        "linear16" | "s16le" => Some(AudioEncoding::Linear16),
        _ => None,
    }
}

fn get_request_value(req: &Request<()>, header_name: &str, query_name: &str) -> Option<String> {
    if let Some(value) = req.headers().get(header_name) {
        if let Ok(value) = value.to_str() {
            return Some(value.to_string());
        }
    }

    let query = req.uri().query()?;
    for (key, value) in form_urlencoded::parse(query.as_bytes()) {
        if key == query_name {
            return Some(value.into_owned());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AppRole, AsrModelProvider};

    fn test_config() -> AppConfig {
        AppConfig {
            role: AppRole::Ingress,
            rust_log: "info".into(),
            log_format: crate::config::LogFormat::Json,
            port: 8443,
            enable_h3: false,
            tls_cert_path: None,
            tls_key_path: None,
            model_dir: Some("/tmp/model".into()),
            model_provider: AsrModelProvider::Cohere,
            device_ids: vec![0],
            onnx_sessions: 1,
            cohere_max_new_tokens: 384,
            chunk_seconds: 30.0,
            overlap_seconds: 2.0,
            final_min_seconds: 0.5,
            utt_split_seconds: 0.8,
            upload_response_num_streams: 16,
            upload_response_slot_size_kb: 32,
            upload_response_slots_per_stream: 1_024,
            upload_response_timeout_ms: 30_000,
            upload_response_watch_poll_ms: 1,
            upload_response_worker_poll_ms: 2,
            upload_response_max_inflight: 2,
            upload_response_worker_id: "test-worker".into(),
            upload_response_ingress_urls: Vec::new(),
            upload_response_discovery_dns: None,
            upload_response_discovery_interval_ms: 2_000,
            upload_response_insecure_tls: false,
            upload_response_worker_heartbeat_interval_ms: 1_000,
            upload_response_worker_ttl_ms: 5_000,
        }
    }

    #[test]
    fn parses_query_flags() {
        let config = test_config();
        let req = Request::builder()
            .uri("/v1/listen?utterances=true&paragraphs=1&utt_split=1.5&language=fr&model=nova&timestamps=true&interim_results=true&encoding=linear16&sample_rate=8000&channels=2&endpointing=250")
            .body(())
            .unwrap();
        let options = ListenOptions::from_request(&req, &config);
        assert!(options.utterances);
        assert!(options.paragraphs);
        assert!(options.timestamps);
        assert!(options.words);
        assert!(options.interim_results);
        assert_eq!(options.utterance_split_secs, 1.5);
        assert_eq!(options.language, "fr");
        assert_eq!(options.model, "nova");
        assert_eq!(options.encoding, Some(AudioEncoding::Linear16));
        assert_eq!(options.sample_rate_hz, Some(8_000));
        assert_eq!(options.channels, 2);
        assert_eq!(options.endpointing_ms, Some(250));
    }

    #[test]
    fn joins_punctuation_without_extra_space() {
        let words = vec![
            CommittedWord {
                index: 0,
                start_ms: 0,
                end_ms: 100,
                word: "hello".into(),
                stitch_start_ms: 0,
                stitch_end_ms: 100,
            },
            CommittedWord {
                index: 1,
                start_ms: 120,
                end_ms: 180,
                word: ",".into(),
                stitch_start_ms: 120,
                stitch_end_ms: 180,
            },
            CommittedWord {
                index: 2,
                start_ms: 220,
                end_ms: 320,
                word: "world".into(),
                stitch_start_ms: 220,
                stitch_end_ms: 320,
            },
        ];
        assert_eq!(transcript_from_words(&words, ""), "hello, world");
    }

    #[test]
    fn request_headers_override_query_for_word_flags() {
        let config = test_config();
        let req = Request::builder()
            .uri("/v1/listen?timestamps=true")
            .header("timestamps", "false")
            .body(())
            .unwrap();
        let options = ListenOptions::from_request(&req, &config);
        assert!(!options.timestamps);
        assert!(!options.words);
    }

    #[test]
    fn defaults_interim_results_off() {
        let config = test_config();
        let req = Request::builder().uri("/v1/listen").body(()).unwrap();
        let options = ListenOptions::from_request(&req, &config);
        assert!(!options.interim_results);
    }
}
