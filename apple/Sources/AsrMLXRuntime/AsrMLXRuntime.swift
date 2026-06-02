import Foundation
import MLX
import MLXNN

public struct ASRTranscriptionResult: Encodable, Sendable {
    public struct Word: Encodable, Sendable {
        public let word: String
        public let startMs: UInt32
        public let endMs: UInt32

        enum CodingKeys: String, CodingKey {
            case word
            case startMs = "start_ms"
            case endMs = "end_ms"
        }
    }

    public let text: String
    public let words: [Word]
    public let tokenIds: [Int]

    enum CodingKeys: String, CodingKey {
        case text
        case words
        case tokenIds = "token_ids"
    }
}

public struct CohereMLXRuntimeSummary: Sendable {
    public let sampleRate: Int
    public let featureCount: Int
    public let nFFT: Int
    public let windowSize: Double
    public let windowStride: Double
    public let vocabSize: Int
    public let supportedLanguages: [String]
    public let loadedTensorCount: Int
}

public enum ASRMLXRuntimeError: Error, LocalizedError, Sendable {
    case missingFile(URL)
    case invalidAudio(String)
    case invalidBundle(String)
    case unsupported(String)

    public var errorDescription: String? {
        switch self {
        case let .missingFile(url):
            return "Missing ASR MLX bundle file: \(url.path)"
        case let .invalidAudio(detail):
            return detail
        case let .invalidBundle(detail):
            return detail
        case let .unsupported(detail):
            return detail
        }
    }
}

public struct CohereModelConfig: Decodable, Sendable {
    public struct Preprocessor: Decodable, Sendable {
        public let sampleRate: Int
        public let features: Int
        public let nFFT: Int
        public let windowSize: Double
        public let windowStride: Double
        public let dither: Double

        enum CodingKeys: String, CodingKey {
            case sampleRate = "sample_rate"
            case features
            case nFFT = "n_fft"
            case windowSize = "window_size"
            case windowStride = "window_stride"
            case dither
        }
    }

    public struct Encoder: Decodable, Sendable {
        public let dModel: Int
        public let nLayers: Int
        public let nHeads: Int

        enum CodingKeys: String, CodingKey {
            case dModel = "d_model"
            case nLayers = "n_layers"
            case nHeads = "n_heads"
        }
    }

    public struct TransformerDecoder: Decodable, Sendable {
        public struct ConfigDict: Decodable, Sendable {
            public let hiddenSize: Int
            public let numLayers: Int
            public let numAttentionHeads: Int
            public let maxSequenceLength: Int

            enum CodingKeys: String, CodingKey {
                case hiddenSize = "hidden_size"
                case numLayers = "num_layers"
                case numAttentionHeads = "num_attention_heads"
                case maxSequenceLength = "max_sequence_length"
            }
        }

        public let configDict: ConfigDict

        enum CodingKeys: String, CodingKey {
            case configDict = "config_dict"
        }
    }

    public let preprocessor: Preprocessor
    public let encoder: Encoder?
    public let transfDecoder: TransformerDecoder?
    public let vocabSize: Int
    public let supportedLanguages: [String]

    enum CodingKeys: String, CodingKey {
        case preprocessor
        case encoder
        case transfDecoder = "transf_decoder"
        case vocabSize = "vocab_size"
        case supportedLanguages = "supported_languages"
    }
}

public struct CohereTokenizer: Sendable {
    public struct SpecialTokens: Sendable {
        public let eos: Int
        public let nospeech: Int
        public let startOfTranscript: Int
        public let pnc: Int
        public let noPnc: Int
        public let startOfContext: Int
        public let noItn: Int
        public let noTimestamp: Int
        public let noDiarize: Int
        public let emoUndefined: Int
        public let languageIds: [String: Int]
        public let specialIds: Set<Int>
    }

    private let idToPiece: [Int: String]
    public let special: SpecialTokens

    public static func load(from modelURL: URL) throws -> CohereTokenizer {
        let vocabURL = modelURL.appendingPathComponent("vocab.json")
        guard FileManager.default.fileExists(atPath: vocabURL.path) else {
            throw ASRMLXRuntimeError.missingFile(vocabURL)
        }
        let rawVocab = try JSONDecoder().decode(
            [String: String].self,
            from: Data(contentsOf: vocabURL)
        )
        var idToPiece = [Int: String]()
        for (key, value) in rawVocab {
            if let id = Int(key) {
                idToPiece[id] = value
            }
        }

        return CohereTokenizer(
            idToPiece: idToPiece,
            special: try SpecialTokens.load(from: modelURL)
        )
    }

    public func prompt(language: String, punctuation: Bool) throws -> [Int] {
        guard let languageId = special.languageIds[language] else {
            throw ASRMLXRuntimeError.invalidBundle("Unsupported Cohere language: \(language)")
        }
        return [
            special.startOfContext,
            special.startOfTranscript,
            special.emoUndefined,
            languageId,
            languageId,
            punctuation ? special.pnc : special.noPnc,
            special.noItn,
            special.noTimestamp,
            special.noDiarize,
        ]
    }

    public func decode(_ ids: [Int]) -> String {
        var result = ""
        for id in ids {
            if special.specialIds.contains(id) {
                continue
            }
            guard let piece = idToPiece[id] else {
                continue
            }
            if piece.hasPrefix("\u{2581}") {
                if !result.isEmpty {
                    result.append(" ")
                }
                result.append(String(piece.dropFirst()))
            } else if piece.hasPrefix("<0x"), piece.hasSuffix(">") {
                let hex = piece.dropFirst(3).dropLast()
                if let byte = UInt8(hex, radix: 16) {
                    let scalar = UnicodeScalar(byte)
                    result.append(Character(scalar))
                }
            } else if piece == "<unk>" || piece == "<s>" || piece == "</s>" {
                continue
            } else {
                result.append(piece)
            }
        }
        return result
    }
}

extension CohereTokenizer.SpecialTokens {
    fileprivate static func load(from modelURL: URL) throws -> CohereTokenizer.SpecialTokens {
        struct TokenEntry: Decodable {
            let content: String
            let special: Bool?
        }
        struct TokenizerConfig: Decodable {
            let addedTokensDecoder: [String: TokenEntry]

            enum CodingKeys: String, CodingKey {
                case addedTokensDecoder = "added_tokens_decoder"
            }
        }

        let configURL = modelURL.appendingPathComponent("tokenizer_config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw ASRMLXRuntimeError.missingFile(configURL)
        }
        let config = try JSONDecoder().decode(
            TokenizerConfig.self,
            from: Data(contentsOf: configURL)
        )

        var tokenToId = [String: Int]()
        var specialIds = Set<Int>()
        for (rawId, entry) in config.addedTokensDecoder {
            guard let id = Int(rawId) else {
                continue
            }
            tokenToId[entry.content] = id
            if entry.special == true {
                specialIds.insert(id)
            }
        }

        func id(_ token: String, fallback: Int? = nil) throws -> Int {
            if let value = tokenToId[token] {
                return value
            }
            if let fallback {
                return fallback
            }
            throw ASRMLXRuntimeError.invalidBundle("Missing Cohere tokenizer special token \(token)")
        }

        let languageCodes = ["en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko"]
        var languageIds = [String: Int]()
        for code in languageCodes {
            if let value = tokenToId["<|\(code)|>"] {
                languageIds[code] = value
                specialIds.insert(value)
            }
        }

        return CohereTokenizer.SpecialTokens(
            eos: try id("<|endoftext|>", fallback: 3),
            nospeech: try id("<|nospeech|>", fallback: 1),
            startOfTranscript: try id("<|startoftranscript|>", fallback: 4),
            pnc: try id("<|pnc|>", fallback: 5),
            noPnc: try id("<|nopnc|>", fallback: 6),
            startOfContext: try id("<|startofcontext|>", fallback: 7),
            noItn: try id("<|noitn|>", fallback: 9),
            noTimestamp: try id("<|notimestamp|>", fallback: 11),
            noDiarize: try id("<|nodiarize|>", fallback: 13),
            emoUndefined: try id("<|emo:undefined|>", fallback: 16),
            languageIds: languageIds,
            specialIds: specialIds
        )
    }
}

public final class CohereMLXRuntime {
    private let modelURL: URL
    private let config: CohereModelConfig
    private let tokenizer: CohereTokenizer
    private let weights: [String: MLXArray]?
    private let graph: CohereMLXGraph?

    public init(modelURL: URL, loadWeights: Bool = true) throws {
        self.modelURL = modelURL.standardizedFileURL.resolvingSymlinksInPath()

        let configURL = self.modelURL.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw ASRMLXRuntimeError.missingFile(configURL)
        }
        self.config = try JSONDecoder().decode(
            CohereModelConfig.self,
            from: Data(contentsOf: configURL)
        )
        self.tokenizer = try CohereTokenizer.load(from: self.modelURL)

        if loadWeights {
            let weightsURL = self.modelURL.appendingPathComponent("model.safetensors")
            guard FileManager.default.fileExists(atPath: weightsURL.path) else {
                throw ASRMLXRuntimeError.missingFile(weightsURL)
            }
            let loadedWeights = try loadArrays(data: Data(contentsOf: weightsURL))
            self.weights = loadedWeights
            self.graph = try CohereMLXGraph(config: self.config, weights: loadedWeights)
        } else {
            self.weights = nil
            self.graph = nil
        }

        guard config.preprocessor.sampleRate == 16_000 else {
            throw ASRMLXRuntimeError.invalidBundle(
                "Cohere MLX runtime expects 16 kHz audio, got \(config.preprocessor.sampleRate)"
            )
        }
    }

    public var summary: CohereMLXRuntimeSummary {
        CohereMLXRuntimeSummary(
            sampleRate: config.preprocessor.sampleRate,
            featureCount: config.preprocessor.features,
            nFFT: config.preprocessor.nFFT,
            windowSize: config.preprocessor.windowSize,
            windowStride: config.preprocessor.windowStride,
            vocabSize: config.vocabSize,
            supportedLanguages: config.supportedLanguages,
            loadedTensorCount: weights?.count ?? 0
        )
    }

    public func transcribe(
        samples: [Float],
        sampleRate: Int,
        maxNewTokens: Int,
        language: String = "en",
        punctuation: Bool = true
    ) throws -> ASRTranscriptionResult {
        guard sampleRate == config.preprocessor.sampleRate else {
            throw ASRMLXRuntimeError.invalidAudio(
                "Cohere MLX runtime expected \(config.preprocessor.sampleRate) Hz PCM, got \(sampleRate) Hz"
            )
        }
        guard !samples.isEmpty else {
            return ASRTranscriptionResult(text: "", words: [], tokenIds: [])
        }

        throw ASRMLXRuntimeError.unsupported(
            "Cohere Swift MLX expects Rust-computed log-mel features; use --features-f32le from the asr-api wrapper."
        )
    }

    public func transcribeFeatures(
        features: [Float],
        featureCount: Int,
        featureSteps: Int,
        maxNewTokens: Int,
        language: String = "en",
        punctuation: Bool = true
    ) throws -> ASRTranscriptionResult {
        guard featureCount == config.preprocessor.features else {
            throw ASRMLXRuntimeError.invalidAudio(
                "Cohere MLX runtime expected \(config.preprocessor.features) features, got \(featureCount)"
            )
        }
        guard featureSteps >= 0, features.count == featureCount * featureSteps else {
            throw ASRMLXRuntimeError.invalidAudio(
                "f32le feature payload has \(features.count) floats, expected \(featureCount * featureSteps)"
            )
        }
        guard featureSteps > 0 else {
            return ASRTranscriptionResult(text: "", words: [], tokenIds: [])
        }
        guard let graph else {
            throw ASRMLXRuntimeError.invalidBundle("Cohere MLX graph was not loaded")
        }

        let prompt = try tokenizer.prompt(language: language, punctuation: punctuation)
        let featureArray = MLXArray(features, [1, featureCount, featureSteps])
        let tokenIDs = try graph.transcribe(
            features: featureArray,
            prompt: prompt,
            eos: tokenizer.special.eos,
            nospeech: tokenizer.special.nospeech,
            maxNewTokens: maxNewTokens
        )
        let text = tokenizer.decode(tokenIDs).trimmingCharacters(in: .whitespacesAndNewlines)
        return ASRTranscriptionResult(text: text, words: [], tokenIds: tokenIDs)
    }
}

public func loadFloat32LittleEndianAudio(from url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    guard data.count % MemoryLayout<Float>.size == 0 else {
        throw ASRMLXRuntimeError.invalidAudio(
            "f32le audio byte length \(data.count) is not divisible by 4"
        )
    }
    return data.withUnsafeBytes { rawBuffer in
        let bytes = rawBuffer.bindMemory(to: UInt8.self)
        var samples = [Float]()
        samples.reserveCapacity(data.count / 4)
        var offset = 0
        while offset < bytes.count {
            let bits =
                UInt32(bytes[offset])
                | (UInt32(bytes[offset + 1]) << 8)
                | (UInt32(bytes[offset + 2]) << 16)
                | (UInt32(bytes[offset + 3]) << 24)
            samples.append(Float(bitPattern: bits))
            offset += 4
        }
        return samples
    }
}
