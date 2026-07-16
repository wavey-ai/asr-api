import AsrMLXRuntime
import Foundation

struct Options {
    var modelDir: URL?
    var audioURL: URL?
    var featuresURL: URL?
    var sampleRate = 16_000
    var featureCount: Int?
    var featureSteps: Int?
    var maxNewTokens = 384
    var checkOnly = false
    var server = false
}

func parseOptions() throws -> Options {
    var options = Options()
    var iterator = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = iterator.next() {
        switch arg {
        case "--model-dir":
            guard let value = iterator.next() else {
                throw ASRMLXRuntimeError.invalidBundle("--model-dir requires a value")
            }
            options.modelDir = URL(fileURLWithPath: value)
        case "--audio-f32le":
            guard let value = iterator.next() else {
                throw ASRMLXRuntimeError.invalidBundle("--audio-f32le requires a value")
            }
            options.audioURL = URL(fileURLWithPath: value)
        case "--features-f32le":
            guard let value = iterator.next() else {
                throw ASRMLXRuntimeError.invalidBundle("--features-f32le requires a value")
            }
            options.featuresURL = URL(fileURLWithPath: value)
        case "--sample-rate":
            guard let value = iterator.next(), let parsed = Int(value) else {
                throw ASRMLXRuntimeError.invalidBundle("--sample-rate requires an integer")
            }
            options.sampleRate = parsed
        case "--feature-count":
            guard let value = iterator.next(), let parsed = Int(value) else {
                throw ASRMLXRuntimeError.invalidBundle("--feature-count requires an integer")
            }
            options.featureCount = parsed
        case "--feature-steps":
            guard let value = iterator.next(), let parsed = Int(value) else {
                throw ASRMLXRuntimeError.invalidBundle("--feature-steps requires an integer")
            }
            options.featureSteps = parsed
        case "--max-new-tokens":
            guard let value = iterator.next(), let parsed = Int(value) else {
                throw ASRMLXRuntimeError.invalidBundle("--max-new-tokens requires an integer")
            }
            options.maxNewTokens = parsed
        case "--check":
            options.checkOnly = true
        case "--server":
            options.server = true
        default:
            throw ASRMLXRuntimeError.invalidBundle("unknown argument \(arg)")
        }
    }
    return options
}

struct ServerRequest: Decodable {
    let featuresF32LE: String
    let featureCount: Int
    let featureSteps: Int

    enum CodingKeys: String, CodingKey {
        case featuresF32LE = "features_f32le"
        case featureCount = "feature_count"
        case featureSteps = "feature_steps"
    }
}

struct ServerReady: Encodable {
    let ready = true
}

struct ServerSuccess: Encodable {
    let ok = true
    let result: ASRTranscriptionResult
}

struct ServerFailure: Encodable {
    let ok = false
    let error: String
}

func writeJSON<T: Encodable>(_ value: T) throws {
    let data = try JSONEncoder().encode(value)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0a]))
}

func runServer(runtime: CohereMLXRuntime, maxNewTokens: Int) throws {
    try writeJSON(ServerReady())
    while let line = readLine() {
        guard let data = line.data(using: .utf8), !data.isEmpty else {
            continue
        }
        do {
            let request = try JSONDecoder().decode(ServerRequest.self, from: data)
            let features = try loadFloat32LittleEndianAudio(
                from: URL(fileURLWithPath: request.featuresF32LE)
            )
            let result = try runtime.transcribeFeatures(
                features: features,
                featureCount: request.featureCount,
                featureSteps: request.featureSteps,
                maxNewTokens: maxNewTokens
            )
            try writeJSON(ServerSuccess(result: result))
        } catch {
            try writeJSON(ServerFailure(error: error.localizedDescription))
        }
    }
}

do {
    let options = try parseOptions()
    guard let modelDir = options.modelDir else {
        throw ASRMLXRuntimeError.invalidBundle("--model-dir is required")
    }

    let runtime = try CohereMLXRuntime(modelURL: modelDir, loadWeights: !options.checkOnly)
    if options.checkOnly {
        let summary = runtime.summary
        let payload: [String: Any] = [
            "sample_rate": summary.sampleRate,
            "features": summary.featureCount,
            "n_fft": summary.nFFT,
            "vocab_size": summary.vocabSize,
            "loaded_tensors": summary.loadedTensorCount,
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0a]))
        exit(0)
    }

    if options.server {
        try runServer(runtime: runtime, maxNewTokens: options.maxNewTokens)
        exit(0)
    }

    if let featuresURL = options.featuresURL {
        guard let featureCount = options.featureCount else {
            throw ASRMLXRuntimeError.invalidBundle("--feature-count is required with --features-f32le")
        }
        guard let featureSteps = options.featureSteps else {
            throw ASRMLXRuntimeError.invalidBundle("--feature-steps is required with --features-f32le")
        }
        let features = try loadFloat32LittleEndianAudio(from: featuresURL)
        let result = try runtime.transcribeFeatures(
            features: features,
            featureCount: featureCount,
            featureSteps: featureSteps,
            maxNewTokens: options.maxNewTokens
        )
        let data = try JSONEncoder().encode(result)
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0a]))
        exit(0)
    }

    guard let audioURL = options.audioURL else {
        throw ASRMLXRuntimeError.invalidBundle("--features-f32le or --audio-f32le is required unless --check is set")
    }
    let samples = try loadFloat32LittleEndianAudio(from: audioURL)
    let result = try runtime.transcribe(
        samples: samples,
        sampleRate: options.sampleRate,
        maxNewTokens: options.maxNewTokens
    )
    let data = try JSONEncoder().encode(result)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0a]))
} catch {
    FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
    exit(1)
}
