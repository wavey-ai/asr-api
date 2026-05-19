# Example use case: Bitneedle scratch tutorial sweep

This example documents a real ASR workload used by the Bitneedle player work:
transcribing public DJ scratching tutorial videos into ASR-derived notes so the
product team can design a scratch-focused sub-UI.

The goal is not to publish bulk verbatim transcripts. The downstream artifact is
a compact research index: detected techniques, high-signal terms, timestamped
summaries, and UI implications.

## System under test

- ASR endpoint: `POST /v1/listen`
- Public test URL: `https://104.105.26.170:18443/v1/listen`
- Instance class: Linode medium Ada GPU instance
- ASR capacity during this run: `2` workers with `2` total sessions
- Client upload concurrency: `4` audio chunks
- Chunk size: `60s`
- Audio normalization: mono `16 kHz` WAV chunks via `ffmpeg`
- Media source path: `av-ingest` resolve and proxy

## Pipeline

1. Discover candidate YouTube scratching tutorials.
2. Resolve each source through `av-ingest`:

```text
GET https://av-proxy.wavey.ai/resolve?url=...
```

3. Select a direct audio format from the `playerResponse`.
4. Download media bytes through the `av-ingest` proxy with ranged `GET`:

```text
GET https://av-proxy.wavey.ai/proxy?url=...
Range: bytes=start-end
```

5. Segment the audio locally:

```bash
ffmpeg -i audio -vn -ac 1 -ar 16000 -acodec pcm_s16le \
  -f segment -segment_time 60 -reset_timestamps 1 chunk_%03d.wav
```

6. Submit each chunk to ASR:

```bash
curl --http2 -k \
  -H 'Content-Type: audio/wav' \
  --data-binary @chunk_000.wav \
  'https://104.105.26.170:18443/v1/listen?utterances=true&paragraphs=true&timestamps=true&language=en_US'
```

7. Convert ASR text into non-verbatim research artifacts:

- scratch technique tags such as `baby`, `chirp`, `flare`, `crab`, `transform`
- UI signals such as fader timing, sound markers, and hand-motion guidance
- timestamped summaries
- aggregate term counts

## Observed throughput

Run date: `2026-05-17`.

The resumed measured portion uploaded `78` chunks across `11` videos. Cached
videos from the previous partial run were excluded from the RTFx calculation.

- Aggregate media duration: `4,378.0s`
- Aggregate summed wall-clock time: `294.4s`
- Observed aggregate RTFx: `14.87x`
- Interpretation: roughly `73.0` minutes of source audio processed in about
  `4.9` minutes of wall time for this workload.

Per-video observed RTFx for newly uploaded chunks:

| Rank | Uploaded chunks | Audio seconds | Wall seconds | RTFx | Video |
| ---: | ---: | ---: | ---: | ---: | --- |
| 6 | 4 | 233.0 | 7.3 | 31.91x | Avionyx Scratch Academy - Episode 6 - The Crab Scratch Tutorial HOW TO |
| 7 | 11 | 660.0 | 33.5 | 19.67x | Why Your Scratching Sounds Bad (And How to Fix It) |
| 8 | 6 | 332.0 | 51.5 | 6.45x | DJ Essentials - Everything You Need to Start Scratching Records |
| 9 | 22 | 1266.0 | 82.5 | 15.34x | Beginner DJ Scratch Tutorial: Tips & Tricks On Getting Started |
| 10 | 19 | 1125.0 | 55.1 | 20.41x | 15 Levels of Turntable Scratching: Easy to Complex |
| 11 | 2 | 80.0 | 8.0 | 10.01x | iphone baby scratch app flare crab autobahn twiddle chirp transform orbit swing flare mpc 2000xl |
| 12 | 4 | 190.0 | 17.8 | 10.69x | Flare Scratch, Transform Scratch, Twiddle, Crab Scratch, Chirp, Orbit |
| 13 | 1 | 35.0 | 7.1 | 4.92x | chirps + flare + crab combo scratch |
| 14 | 1 | 35.0 | 3.7 | 9.42x | SCRATCH PRACTICE - Chirp Flare / 3 Click Flare (Crab) |
| 15 | 1 | 60.0 | 4.4 | 13.64x | Ortofon Scratch Tutorial #13: Variation of the chirp flare |
| 16 | 7 | 362.0 | 23.4 | 15.48x | Learn How to Scratch: The Chirp Flare Scratch (Tutorial 17) |

## Why this is a useful ASR benchmark

This workload is representative of product research ingestion rather than a
synthetic single-file benchmark:

- mixed source durations from short clips to long tutorials
- real internet media retrieval through `av-ingest`
- chunk-level parallelism
- conversational tutorial audio with music, scratching, and equipment noise
- post-ASR extraction into structured product-design notes

It exercises endpoint latency, chunk throughput, worker scheduling, and practical
failure handling around media resolution.

## Notes for future runs

- Keep source media retrieval in `av-ingest`, not direct `yt-dlp`, when testing
  the production media path.
- Prefer `ANDROID_VR` YouTube resolution with `visitorData`; the older Android
  resolver can produce signed URLs that fail later byte ranges.
- Store summaries and technique maps for third-party videos rather than full
  verbatim transcripts.
- Report RTFx only for chunks actually uploaded to ASR. Cached chunks should be
  counted separately.
