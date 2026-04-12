use crate::config::AppConfig;
use anyhow::Result;
use bytes::Bytes;
use futures_util::StreamExt;
use http::{header::CONTENT_TYPE, Request, StatusCode};
use http_pack::stream::StreamHeaders;
use serde_json::json;
use soundkit::audio_pipeline::{deserialize_audio, vec_i16_to_f32, vec_i32_to_f32};
use soundkit::audio_types::{AudioData, PcmData};
use soundkit_decoder::{DecodeOptions, DecodePipeline, DecodePipelineHandle};
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio::time::{timeout, Duration};
use upload_response::{ResponseResult, UploadResponseService};
use web_service::{BodyStream, HandlerResponse};

#[derive(Clone)]
pub struct ListenIngress {
    config: AppConfig,
    service: Arc<UploadResponseService>,
}

impl ListenIngress {
    pub fn new(config: AppConfig, service: Arc<UploadResponseService>) -> Self {
        Self { config, service }
    }

    pub async fn handle_listen(&self, req: Request<()>, body: BodyStream) -> HandlerResponse {
        match self.handle_listen_inner(req, body).await {
            Ok(response) => response,
            Err(error) => error_response(classify_error(&error), error.to_string()),
        }
    }

    async fn handle_listen_inner(
        &self,
        req: Request<()>,
        body: BodyStream,
    ) -> Result<HandlerResponse> {
        reject_json_requests(&req)?;

        let stream = self
            .service
            .open_stream()
            .await
            .map_err(anyhow::Error::msg)?;
        let stream_id = stream.stream_id();
        let rx = self.service.register_response(stream_id).await;

        let result = async {
            let headers = StreamHeaders::from_request(stream_id, &req)
                .map_err(|error| anyhow::anyhow!("failed to encode request headers: {error}"))?;
            self.service
                .write_request_headers(stream_id, headers)
                .await
                .map_err(anyhow::Error::msg)?;

            self.transcode_request_body(stream_id, body).await?;
            self.service
                .end_request(stream_id)
                .await
                .map_err(anyhow::Error::msg)?;

            self.await_response(stream_id, rx).await
        }
        .await;

        stream.close().await;
        result
    }

    async fn transcode_request_body(&self, stream_id: u64, mut body: BodyStream) -> Result<()> {
        let mut decoder = Self::new_decoder();
        let mut received_bytes = 0usize;

        while let Some(next) = body.next().await {
            let chunk =
                next.map_err(|error| anyhow::anyhow!("failed to read request body: {error}"))?;
            if chunk.is_empty() {
                continue;
            }

            received_bytes += chunk.len();
            self.send_decoder_bytes(&mut decoder, chunk).await?;
            self.flush_decoder(stream_id, &mut decoder, false).await?;
        }

        anyhow::ensure!(
            received_bytes > 0,
            "request body did not include audio bytes"
        );

        self.send_decoder_bytes(&mut decoder, Bytes::new()).await?;
        self.flush_decoder(stream_id, &mut decoder, true).await?;
        Ok(())
    }

    async fn send_decoder_bytes(
        &self,
        decoder: &mut DecodePipelineHandle,
        data: Bytes,
    ) -> Result<()> {
        loop {
            match decoder.send(data.clone()) {
                Ok(()) => return Ok(()),
                Err(soundkit_decoder::DecodeError::InputBufferFull) => {
                    tokio::task::yield_now().await;
                }
                Err(error) => {
                    return Err(anyhow::anyhow!("decoder send failed: {error}"));
                }
            }
        }
    }

    async fn flush_decoder(
        &self,
        stream_id: u64,
        decoder: &mut DecodePipelineHandle,
        blocking: bool,
    ) -> Result<()> {
        if blocking {
            while let Some(output) = decoder.recv() {
                let audio = output.map_err(|error| anyhow::anyhow!("decode failed: {error}"))?;
                self.append_audio(stream_id, &audio).await?;
            }
            return Ok(());
        }

        while let Some(output) = decoder.try_recv() {
            let audio = output.map_err(|error| anyhow::anyhow!("decode failed: {error}"))?;
            self.append_audio(stream_id, &audio).await?;
        }
        Ok(())
    }

    async fn append_audio(&self, stream_id: u64, audio: &AudioData) -> Result<()> {
        let samples = audio_to_mono_f32(audio)?;
        if samples.is_empty() {
            return Ok(());
        }

        let mut pcm_bytes = Vec::with_capacity(samples.len() * std::mem::size_of::<f32>());
        for sample in samples {
            pcm_bytes.extend_from_slice(&sample.to_le_bytes());
        }

        let slot_bytes = self.config.upload_response_config().slot_bytes().max(1);
        for chunk in pcm_bytes.chunks(slot_bytes) {
            self.service
                .append_request_body(stream_id, Bytes::copy_from_slice(chunk))
                .await
                .map_err(anyhow::Error::msg)?;
        }

        Ok(())
    }

    async fn await_response(
        &self,
        stream_id: u64,
        rx: oneshot::Receiver<ResponseResult>,
    ) -> Result<HandlerResponse> {
        let timeout_duration = Duration::from_millis(self.config.upload_response_timeout_ms);
        match timeout(timeout_duration, rx).await {
            Ok(Ok(Ok(cached))) => {
                let mut content_type = None;
                let mut headers = Vec::new();
                for (name, value) in cached.headers {
                    if name.eq_ignore_ascii_case("content-type") {
                        content_type = Some(value);
                    } else {
                        headers.push((name, value));
                    }
                }

                Ok(HandlerResponse {
                    status: cached.status,
                    body: Some(cached.body),
                    content_type,
                    headers,
                    etag: None,
                })
            }
            Ok(Ok(Err(error))) => {
                self.service.drop_response_channel(stream_id).await;
                Err(anyhow::anyhow!(error))
            }
            Ok(Err(_)) => {
                self.service.drop_response_channel(stream_id).await;
                Err(anyhow::anyhow!("response channel closed"))
            }
            Err(_) => {
                self.service.drop_response_channel(stream_id).await;
                Err(anyhow::anyhow!("response timeout"))
            }
        }
    }

    fn new_decoder() -> DecodePipelineHandle {
        DecodePipeline::spawn_with_options(DecodeOptions {
            output_bits_per_sample: Some(16),
            output_sample_rate: Some(crate::config::ASR_SAMPLE_RATE),
            output_channels: Some(1),
        })
    }
}

fn audio_to_mono_f32(audio: &AudioData) -> Result<Vec<f32>> {
    let channels: Vec<Vec<f32>> =
        match deserialize_audio(audio.data(), audio.bits_per_sample(), audio.channel_count())
            .map_err(|error| anyhow::anyhow!("failed to deserialize PCM: {error}"))?
        {
            PcmData::I16(channels) => channels.into_iter().map(vec_i16_to_f32).collect(),
            PcmData::I32(channels) => channels.into_iter().map(vec_i32_to_f32).collect(),
            PcmData::F32(channels) => channels,
        };

    if channels.is_empty() {
        return Ok(Vec::new());
    }
    if channels.len() == 1 {
        return Ok(channels.into_iter().next().unwrap_or_default());
    }

    let len = channels[0].len();
    let mut mono = vec![0.0f32; len];
    for channel in &channels {
        for (index, sample) in channel.iter().enumerate().take(len) {
            mono[index] += *sample;
        }
    }

    let scale = 1.0 / channels.len() as f32;
    for sample in &mut mono {
        *sample *= scale;
    }
    Ok(mono)
}

fn reject_json_requests(req: &Request<()>) -> Result<()> {
    let is_json = req
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            let value = value.to_ascii_lowercase();
            value.starts_with("application/json") || value.ends_with("+json")
        })
        .unwrap_or(false);

    if is_json {
        anyhow::bail!(
            "JSON URL payloads are not supported yet; upload audio bytes in the request body"
        );
    }

    Ok(())
}

fn classify_error(error: &anyhow::Error) -> StatusCode {
    let message = error.to_string().to_ascii_lowercase();
    if message.contains("request body")
        || message.contains("not supported")
        || message.contains("decode")
        || message.contains("deserialize")
        || message.contains("audio bytes")
    {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

fn error_response(status: StatusCode, message: String) -> HandlerResponse {
    HandlerResponse {
        status,
        body: Some(Bytes::from(
            serde_json::to_vec(&json!({ "error": message }))
                .unwrap_or_else(|_| b"{\"error\":\"serialization failure\"}".to_vec()),
        )),
        content_type: Some("application/json".into()),
        headers: vec![("cache-control".into(), "no-store".into())],
        etag: None,
    }
}
