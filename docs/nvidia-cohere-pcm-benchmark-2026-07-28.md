# NVIDIA Cohere PCM Benchmark: July 28, 2026

## Purpose

This report records a full-corpus Cohere Transcribe benchmark on one NVIDIA host.
It measures throughput, resource use, model timing, and service-path overhead.

Real-time factor (`RTFx`) is audio duration divided by processing duration.
A larger `RTFx` value indicates more throughput.

The test used cached raw PCM.
Thus, the result does not include compressed-audio decoding.

## Result

The run processed all `249` sources without a failure.
The sources contained `152,791` seconds, or `42.442` hours, of audio.

The benchmark completed in `1,949.620` seconds.
Its effective throughput was `78.370x` realtime.

The complete process used `2,026.171` seconds.
This value includes model load, TensorRT cache load, test setup, and shutdown.
The lifecycle-inclusive throughput was `75.409x` realtime.

| Result | Value |
| --- | ---: |
| Completed sources | `249` |
| Failed sources | `0` |
| Audio duration | `152,791 s` |
| Benchmark duration | `1,949.620 s` |
| Process duration | `2,026.171 s` |
| Effective throughput | `78.370x` |
| Lifecycle-inclusive throughput | `75.409x` |
| Summed ASR service duration | `5,837.839 s` |
| Summed ASR service throughput | `26.173x` |
| Average active ASR concurrency | `2.994` |

The average active concurrency was `99.8%` of the configured limit.
The orchestration kept the three-request queue full.

## RTFx Definitions

This report uses three `RTFx` values.
Each value answers a different capacity question.

### ASR Service RTFx

ASR service `RTFx` divides total audio duration by summed source ASR duration.
The result was `26.173x`.

This value includes contention between the three active requests.
It describes the measured service rate for one active request in that workload.

Do not use the unweighted mean source `RTFx` for fleet capacity.
Short sources have too much influence on that mean.

### Effective RTFx

Effective `RTFx` divides total audio duration by benchmark elapsed duration.
The result was `78.370x`.

This value includes the throughput gain from concurrent requests.
It excludes process setup and shutdown.

The following relation explains the measured result:

```text
26.173 service RTFx × 2.994 average active requests = 78.362 effective RTFx
```

The small difference comes from rounding.

### Lifecycle-Inclusive RTFx

Lifecycle-inclusive `RTFx` divides total audio duration by complete process duration.
The result was `75.409x`.

This value includes warm TensorRT cache load and service setup.
Use it for isolated jobs that start a new process.

Use effective `RTFx` for a continuously warm service.
Use lifecycle-inclusive `RTFx` for a process-per-job design.

### Interim Metric Correction

An interim status update reported approximately `119x`.
That value was not a valid `RTFx`.

The calculation divided summed completed audio by the longest single-source duration.
It mixed an aggregate numerator with a per-source denominator.

At that checkpoint, the run had completed `20,585` audio seconds.
The correct cumulative elapsed duration was `308.165` seconds.
The correct checkpoint rate was `66.799x`.

The highest valid cumulative rate was `78.409x` after `247` sources.
The final rate was `78.370x` after all `249` sources.

## Batch Capacity

The measured warm batch capacity was `78.370` audio hours for each wall-clock hour.
This value equals `1,880.9` audio hours for each day.

Capacity planning must reserve operating margin.
The following table applies simple utilization limits to the measured result.

| Target utilization | Audio hours/hour | Audio hours/day |
| --- | ---: | ---: |
| `100%` measured rate | `78.370` | `1,880.9` |
| `80%` planning rate | `62.696` | `1,504.7` |
| `70%` planning rate | `54.859` | `1,316.6` |

For a warm worker, use this estimate:

```text
wall duration = audio duration / 78.370
```

| Batch audio | Estimated warm wall duration |
| --- | ---: |
| `1` audio hour | `45.9 s` |
| `10` audio hours | `7.66 min` |
| `100` audio hours | `76.56 min` |

These values describe batch throughput.
They do not prove support for `78` simultaneous live connections.

The test used only three concurrent requests.
It also supplied cached audio faster than realtime.

Run a separate connection test for live-stream capacity.
That test must include client pacing, network transport, and connection memory.

## Corpus Metadata

The test identified the corpus by metadata and a deterministic fingerprint.
It did not use the collection name as a test identifier.

| Property | Value |
| --- | --- |
| Dataset fingerprint | `2ccba722531c557cce185244454fa3ad9d6b9b366fd6852d3d6b709c69b13770` |
| Source count | `249` |
| Audio duration | `152,791 s` |
| Audio format | Headerless signed `16-bit` little-endian PCM |
| Sample rate | `16 kHz` |
| Channels | `1` |
| PCM bytes | `4,888,974,368` |
| PCM size | `4.553 GiB` |

The source duration distribution was not uniform.
Long sources dominated the total audio duration.

| Source duration | Seconds |
| --- | ---: |
| Minimum | `21` |
| Mean | `613.618` |
| P50 | `85` |
| P90 | `3,521` |
| P95 | `4,232.2` |
| P99 | `4,926.16` |
| Maximum | `6,139` |

FFmpeg `6.1.1` created the PCM files before this run.
The conversion used mono `16 kHz` signed `16-bit` little-endian output.

All converted files passed size, metadata, and duration checks.
The maximum source-to-PCM duration difference was `0.028` seconds.

## Host

| Component | Value |
| --- | --- |
| Provider | Linode |
| GPU | NVIDIA RTX 4000 Ada Generation |
| GPU memory | `20,475 MiB` |
| Compute capability | `8.9` |
| NVIDIA driver | `570.211.01` |
| CUDA compatibility | `12.8` |
| TensorRT | `10.9.0.34` |
| ONNX Runtime | `1.23.2` GPU package |
| CPU | AMD EPYC 9474F |
| Logical CPUs | `8` |
| Host memory | `33,653,866,496` bytes |
| Operating system | Ubuntu `24.04` |
| Kernel | Linux `6.8.0-134-generic` |

## Software Revisions

The full run used these principal revisions:

| Repository | Revision |
| --- | --- |
| `asr-api` | `1789589df1aba417724253728f7050c88c715d08` |
| `media-research-stack` | `3434b6802c57ad93511139003417de73594ba589` |
| `web-services` | `ab42087605a8f06ee2a92cf90b85ea30f262d642` |
| `gpu-workers` | `58ac4697deee880c4f9596bb4ea21cac8d1fc542` |
| `av-api` | `6b50d0025adfa924a6c2d03338fde663244cb10c` |
| `av-ingest` | `19b2c085a10377e28c617be12d2977c5eb562658` |

Revision `cc46231` later corrected the benchmark token limit.
See [Token Limit Finding](#token-limit-finding).

## Runtime Configuration

The run used one ASR worker.
The worker owned two ONNX sessions and accepted three concurrent requests.

| Setting | Value |
| --- | --- |
| Model | Cohere Transcribe 03-2026 |
| Backend | ONNX Runtime |
| Execution provider | TensorRT |
| TensorRT components | `all` |
| TensorRT precision | FP16 |
| ONNX sessions | `2` |
| ASR concurrency | `3` |
| Worker instances | `1` |
| Window duration | `30 s` |
| Window overlap | `2 s` |
| Timestamp backend | `token-frequency` |
| Maximum new tokens | `128` |
| TensorRT minimum feature frames | `20` |
| TensorRT optimum duration | `30 s` |
| TensorRT maximum duration | `35 s` |

The test used a `20`-frame TensorRT minimum.
The smallest observed model input contained `81` feature frames.
TensorRT did not reject a final input.

### Request Handoff

The in-process test used the complete ingress, decoder, worker, and response path.
The path used `upload-response` for stage handoff.

| Setting | Value |
| --- | ---: |
| Concurrent streams | `3` |
| Maximum in-flight claims | `3` |
| Slot size | `32 KiB` |
| Ring bytes for each stream | `64 MiB` |
| Derived slots for each stream | `2,048` |
| Watch poll interval | `1 ms` |
| Worker poll interval | `2 ms` |
| Request timeout | `21,600,000 ms` |

The harness read cached PCM in `64 KiB` blocks.
The service converted that PCM to canonical `f32` samples.

### Ring Capacity

Mono `16 kHz` signed `16-bit` PCM uses `31.25 KiB/s`.
A `64 MiB` ring can contain `2,097.152` seconds of this PCM.
This duration is approximately `34.95` minutes.

The three configured streams had `192 MiB` of total logical ring capacity.
Each stream used `2,048` slots of `32 KiB`.

The largest source contained `187.347 MiB` of PCM.
It was larger than one complete ring.

The source completed because the reader released consumed slots.
Thus, ring capacity is not a maximum source-size limit.

The test did not record the ring high-water mark.
It also did not record how long a writer waited for a slot.

These missing values prevent an exact buffer-capacity margin.
Add the metrics in [Recommended Metrics](#recommended-metrics).

## Transcript Output

The run saved one transcript for each source.
No transcript was empty.

| Output | Value |
| --- | ---: |
| Transcript files | `249` |
| Words | `461,807` |
| Characters | `2,386,129` |
| Empty transcripts | `0` |
| Transcripts below five words | `1` |
| Words for each audio second | `3.0225` |
| Words for each ASR service second | `79.106` |

This report does not measure transcription accuracy.
A separate comparison must use exact Unicode character edit distance.

## macOS-to-NVIDIA Edit Distance

The comparison paired all `249` macOS and NVIDIA transcript files by source.
It used exact Levenshtein distance over stored Unicode code points.

The calculation did not remove punctuation.
It did not fold case, normalize Unicode, or collapse whitespace.
It summed each source distance without joining source boundaries.

| Exact comparison | All pairs | Definite valid intersection |
| --- | ---: | ---: |
| Source pairs | `249` | `231` |
| Excluded macOS sources | `0` | `18` |
| Identical pairs | `1` | `1` |
| macOS characters | `1,251,386` | `1,199,529` |
| NVIDIA characters | `2,388,822` | `1,322,185` |
| Total edit distance | `1,257,631` | `240,913` |
| Distance / larger character total | `52.646%` | `18.221%` |
| P50 source distance | `92` | `88` |
| P95 source distance | `51,135.4` | `8,992` |
| Maximum source distance | `86,902` | `17,905` |

The full comparison includes `18` definitely incomplete macOS sources.
Those sources make the full distance unsuitable as a model-quality score.

The valid intersection excludes those `18` sources.
Its raw edit distance is `240,913` Unicode code points.

This metric is edit distance, not word error rate.
It does not determine which transcript is correct.

See
[edit-distance-osx-vs-nvidia.json](evidence/nvidia-cohere-pcm-2026-07-28/edit-distance-osx-vs-nvidia.json)
for the per-source values.

## Source Throughput

Per-source `ASR RTFx` uses only the time inside the ASR stream call.
Per-source pipeline `RTFx` also includes cache-open and transcript-write work.

| Metric | Mean | P50 | P90 | P95 | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ASR `RTFx` | `29.340` | `28.935` | `33.859` | `34.952` | `43.192` | `56.126` |
| Pipeline `RTFx` | `29.332` | `28.927` | `33.852` | `34.936` | `43.184` | `56.004` |
| PCM input, MiB/s | `0.895` | `0.883` | `1.029` | `1.061` | `1.316` | `1.716` |
| Transcript words/s | `74.706` | `75.106` | `85.015` | `87.695` | `95.801` | `131.040` |

The mean source ASR duration was `23.445` seconds.
The P50 duration was `2.854` seconds.
The P99 duration was `203.023` seconds.

## GPU Measurements

The monitor collected `4,050` samples at `500 ms` intervals.
These samples cover process setup, the benchmark, and shutdown.

| GPU metric | Mean | P50 | P90 | P95 | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Utilization, percent | `78.850` | `82` | `87` | `87` | `90` | `93` |
| Memory, MiB | `9,397.257` | `9,662` | `9,662` | `9,662` | `9,662` | `9,662` |
| Temperature, Celsius | `59.503` | `61` | `63` | `63` | `64` | `64` |
| Power, watts | `79.084` | `81.150` | `83.400` | `84.120` | `85.770` | `88.750` |

The stable GPU allocation was `9.436 GiB`.
This allocation was below half of the installed GPU memory.

The run did not show thermal or memory pressure.
GPU utilization remained below full saturation.

## Model Timing

The timing log contained `29,470` Cohere model invocations.
The model usually received `600` feature steps.

| Invocation metric | Mean | P50 | P90 | P95 | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Total, ms | `131.447` | `130.610` | `176.630` | `190.971` | `221.827` | `707.760` |
| Encoder run, ms | `25.337` | `24.170` | `25.520` | `38.890` | `40.373` | `157.780` |
| Decoder prefill run, ms | `3.309` | `3.200` | `3.720` | `4.030` | `6.910` | `29.770` |
| Cached decoder input, ms | `9.828` | `9.570` | `14.470` | `16.100` | `19.713` | `98.400` |
| Cached decoder run, ms | `88.404` | `88.300` | `126.990` | `139.025` | `164.799` | `522.760` |
| Cached decoder extract, ms | `3.625` | `3.410` | `5.770` | `6.635` | `8.600` | `58.760` |

The instrumented model invocations used `3,873.753` summed seconds.
They represented `66.356%` of summed ASR service time.

The remaining service time includes mel creation, session waits, and stream coordination.
The current metrics cannot divide that time further.

### Timing Share

| Component | Summed time | Share of invocation time |
| --- | ---: | ---: |
| Cached decoder run | `2,605.262 s` | `67.25%` |
| Encoder run | `746.686 s` | `19.28%` |
| Cached decoder input | `289.619 s` | `7.48%` |
| Cached decoder extract | `106.823 s` | `2.76%` |
| Decoder prefill run | `97.516 s` | `2.52%` |
| Decoder prefill extract | `16.776 s` | `0.43%` |
| Encoder extract | `2.601 s` | `0.07%` |
| Token decode | `2.024 s` | `0.05%` |

The cached autoregressive decoder was the largest measured cost.
The encoder was the second-largest measured cost.

Additional worker processes would load more model state.
They would not remove either model cost.

## Service-Path Assessment

The run kept `2.994` requests active on average.
The configured limit was `3`.

This result shows that the request scheduler supplied continuous work.
Worker starvation was not a material limit.

The summed pipeline duration exceeded summed ASR duration by `0.212` seconds.
The mean harness overhead was `0.850 ms` for each source.
The P99 harness overhead was `2.150 ms`.

This harness overhead excludes work inside the ASR stream call.
It does not isolate the `upload-response` implementation.

The run had no stream failure, timeout, or TensorRT profile rejection.
Thus, the current handoff settings supported this workload.

The test did not record ring occupancy or writer wait duration.
Therefore, it cannot prove that the buffer has the optimum size.

## TensorRT Cache

The TensorRT cache used `4,517,910,487` bytes, or `4.208 GiB`.

| Cache item | Bytes |
| --- | ---: |
| Encoder engine | `3,928,407,444` |
| Decoder prefill engine | `308,873,524` |
| Cached decoder engine | `272,546,156` |
| Timing cache | `8,081,954` |

A cold ten-source gate required `626.697` process seconds.
The measured transcription part required only `51.664` seconds.

Cold engine creation added approximately `575.033` seconds.
The warm full run added `76.551` lifecycle seconds.

Keep the service process warm for short or latency-sensitive jobs.
Reuse a cache only with a compatible model and profile.

## Token Limit Finding

The benchmark harness used a `128`-token limit for each ASR window.
The production `asr-api` default is `384`.

Six of `29,470` model invocations reached exactly `128` tokens.
Those invocations can contain truncated dense speech.

The old log did not attach a source identifier to each model invocation.
It mapped each capped invocation to three active requests.

The union contains `17` possible sources.
A follow-up run used `384` tokens for all `17` sources.

Revision `cc46231` changed the benchmark default to `384`.
The revision also changed the recorded benchmark setting.

Use `ASR_COHERE_MAX_NEW_TOKENS=384` for production-equivalent tests.
Log the source and window identifiers with future model timings.

### Corrective Run

The corrective run completed all `17` candidate sources without a failure.
It processed `44,536` audio seconds in `627.503` benchmark seconds.

| Corrective result | Value |
| --- | ---: |
| Completed sources | `17` |
| Failed sources | `0` |
| Audio duration | `44,536 s` |
| Benchmark duration | `627.503 s` |
| Effective throughput | `70.973x` |
| ASR service throughput | `24.555x` |
| Maximum GPU memory | `9,858 MiB` |
| Maximum new tokens | `384` |

Four candidate transcripts were byte-identical to the first run.
Thirteen candidate transcripts changed.

Six model invocations produced more than `128` tokens.
Their token counts were `183`, `231`, `287`, `384`, `384`, and `384`.

Three invocations reached the production limit of `384`.
This result needs source-level inspection before another limit increase.

A limit stop can indicate dense speech, repetition, or a missing end token.
Do not increase the limit without checking the affected window text.

The local composite contains the `17` rerun transcripts and `232` original transcripts.
It keeps one transcript for each of the `249` sources.

## Bottleneck Assessment

The measured hot path is model inference.
The cached decoder run used `67.25%` of measured model time.
The encoder used another `19.28%`.

The orchestration held the configured concurrency limit.
Increasing the worker count is not the next optimization.

Cached decoder input preparation used `7.48%` of measured model time.
Preallocated tensors or ONNX I/O binding can reduce this cost.
Measure allocation and copy counts before this change.

GPU utilization had a median of `82%`.
Autoregressive decoder dependencies prevent simple full-GPU saturation.

The service path showed no failure or visible backpressure.
Direct buffer metrics are still necessary for a final buffer decision.

## Recommended Metrics

Add these `upload-response` metrics:

- ring occupancy and high-water mark for each stream;
- writer wait count and writer wait duration;
- reader lag in slots and bytes;
- claim-to-first-byte duration;
- final-stage-to-response duration;
- bytes copied for each stage;
- active and queued claim counts.

Add these ASR metrics:

- source and window identifiers on each timing record;
- mel-frontend duration;
- ONNX session wait duration;
- TensorRT engine load duration;
- token-limit stop count;
- final-tail feature-frame count.

These metrics will separate GPU work, CPU work, queue waits, and buffer waits.

## Limitations

This was one full run.
The report does not contain run-to-run variance.

The input was raw PCM.
The result does not measure WebM, Opus, or another codec.

The transport was in-process.
The result does not measure network latency or remote TLS.

The run used approximate token-frequency timestamps.
It does not measure side-model timestamp quality or cost.

The run used the benchmark-specific `128`-token limit.
The follow-up test corrects that value for the affected sources.
Three follow-up invocations also reached the `384`-token limit.

The report does not measure transcript accuracy.
Use exact Unicode character edit distance for the planned platform comparison.

## Conclusion

The current one-worker orchestration supplied work at the configured limit.
The principal measured limit was Cohere model execution on the GPU.

The cached decoder is the first model optimization target.
Cached input preparation is the first supporting-code target.

The current request buffer handled the workload without an error.
Additional occupancy and wait metrics are required before a final size claim.

Keep TensorRT workers warm.
Do not add worker processes without a measured session or queue benefit.
