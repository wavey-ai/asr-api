import Foundation
import MLX
import MLXNN

final class CohereMLXGraph {
    private let encoder: CohereConformerEncoder
    private let decoder: CohereTransformerDecoder

    init(config: CohereModelConfig, weights: [String: MLXArray]) throws {
        let store = CohereWeightStore(weights)
        self.encoder = try CohereConformerEncoder(config: config, weights: store)
        self.decoder = try CohereTransformerDecoder(config: config, weights: store)
    }

    func transcribe(features: MLXArray, prompt: [Int], eos: Int, nospeech: Int, maxNewTokens: Int)
        throws -> [Int]
    {
        let encoderInput = envFlag("ASR_COHERE_MLX_F32_ENCODER") ? features : features.asType(.bfloat16)
        let encoderHidden = try encoder.forward(encoderInput)
        try dumpArray(encoderHidden, envName: "ASR_COHERE_MLX_DUMP_ENCODER")
        let crossKV = try decoder.precomputeCrossKV(encoderHidden)
        if let first = crossKV.first {
            try dumpArray(first.key, envName: "ASR_COHERE_MLX_DUMP_CROSS_KEY0")
        }
        var selfKV = Array(
            repeating: CohereSelfKV(key: nil, value: nil),
            count: decoder.layerCount
        )

        var logits: MLXArray?
        for (index, tokenID) in prompt.enumerated() {
            let step = try decoder.step(
                tokenID: tokenID,
                position: index,
                selfKV: selfKV,
                crossKV: crossKV
            )
            logits = step.logits
            selfKV = step.selfKV
        }
        if let first = selfKV.first, let key = first.key {
            try dumpArray(key, envName: "ASR_COHERE_MLX_DUMP_SELF_KEY0")
        }

        guard var lastLogits = logits else {
            return []
        }

        var generated = [Int]()
        generated.reserveCapacity(maxNewTokens)
        var nextToken = argmaxToken(lastLogits)
        let positionOffset = envInt("ASR_COHERE_MLX_POSITION_OFFSET") ?? 0
        var position = prompt.count + positionOffset

        while generated.count < maxNewTokens {
            if nextToken == eos || nextToken == nospeech {
                break
            }
            generated.append(nextToken)

            let step = try decoder.step(
                tokenID: nextToken,
                position: position,
                selfKV: selfKV,
                crossKV: crossKV
            )
            lastLogits = step.logits
            selfKV = step.selfKV
            position += 1
            nextToken = argmaxToken(lastLogits)
        }

        if envFlag("ASR_COHERE_MLX_DEBUG_TOKENS") {
            stderrLine("cohere_mlx_tokens=\(generated)")
        }
        return generated
    }
}

private func envFlag(_ name: String) -> Bool {
    switch ProcessInfo.processInfo.environment[name] {
    case "1", "true", "TRUE", "True", "yes", "YES":
        return true
    default:
        return false
    }
}

private func envInt(_ name: String) -> Int? {
    ProcessInfo.processInfo.environment[name].flatMap(Int.init)
}

private func stderrLine(_ text: String) {
    FileHandle.standardError.write(Data((text + "\n").utf8))
}

private func dumpArray(_ array: MLXArray, envName: String) throws {
    guard let path = ProcessInfo.processInfo.environment[envName], !path.isEmpty else {
        return
    }
    try dumpArray(array, path: path, label: envName)
}

private func dumpArray(_ array: MLXArray, path: String, label: String) throws {
    let dumped = array.asType(.float32).asData(access: .copy)
    try dumped.data.write(to: URL(fileURLWithPath: path))
    stderrLine("cohere_mlx_dump label=\(label) path=\(path) shape=\(array.shape)")
}

private final class CohereWeightStore {
    private let weights: [String: MLXArray]
    private let quantization: CohereLinearQuantization?

    init(_ weights: [String: MLXArray]) {
        self.weights = weights
        self.quantization = CohereLinearQuantization.fromEnvironment()
    }

    func tensor(_ name: String) throws -> MLXArray {
        guard let tensor = weights[name] else {
            throw ASRMLXRuntimeError.invalidBundle("Missing Cohere MLX weight \(name)")
        }
        if envFlag("ASR_COHERE_MLX_F32_WEIGHTS"), tensor.dtype == .bfloat16 {
            return tensor.asType(.float32)
        }
        return tensor
    }

    func linear(_ name: String) throws -> CohereLinearWeight {
        let weight = try tensor(name)
        return try CohereLinearWeight(name: name, weight: weight, quantization: quantization)
    }
}

private struct CohereLinearQuantization {
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode

    static func fromEnvironment() -> CohereLinearQuantization? {
        let env = ProcessInfo.processInfo.environment
        guard let rawBits = env["ASR_COHERE_MLX_QUANT_BITS"]?.trimmingCharacters(in: .whitespaces),
            !rawBits.isEmpty,
            let bits = Int(rawBits),
            bits > 0
        else {
            return nil
        }
        let groupSize = Int(env["ASR_COHERE_MLX_QUANT_GROUP_SIZE"] ?? "") ?? 64
        let modeRaw = (env["ASR_COHERE_MLX_QUANT_MODE"] ?? "affine").lowercased()
        let mode = QuantizationMode(rawValue: modeRaw) ?? .affine
        return CohereLinearQuantization(groupSize: groupSize, bits: bits, mode: mode)
    }
}

private final class CohereLinearWeight {
    private let weight: MLXArray
    private let quantizedWeight: MLXArray?
    private let scales: MLXArray?
    private let quantBiases: MLXArray?
    private let groupSize: Int?
    private let bits: Int?
    private let mode: QuantizationMode

    init(name: String, weight: MLXArray, quantization: CohereLinearQuantization?) throws {
        self.weight = weight
        if let quantization {
            guard weight.shape.count == 2 else {
                throw ASRMLXRuntimeError.invalidBundle(
                    "Cohere MLX linear weight \(name) is not rank-2: \(weight.shape)"
                )
            }
            guard weight.shape[0] % 32 == 0, weight.shape[1] % 32 == 0 else {
                throw ASRMLXRuntimeError.invalidBundle(
                    "Cohere MLX linear weight \(name) shape \(weight.shape) cannot use MLX quantizedMM; dimensions must be multiples of 32"
                )
            }
            guard weight.shape[1] % quantization.groupSize == 0 else {
                throw ASRMLXRuntimeError.invalidBundle(
                    "Cohere MLX linear weight \(name) input dimension \(weight.shape[1]) is not divisible by quant group size \(quantization.groupSize)"
                )
            }
            let quantized = MLX.quantized(
                weight,
                groupSize: quantization.groupSize,
                bits: quantization.bits,
                mode: quantization.mode
            )
            self.quantizedWeight = quantized.wq
            self.scales = quantized.scales
            self.quantBiases = quantized.biases
            self.groupSize = quantization.groupSize
            self.bits = quantization.bits
            self.mode = quantization.mode
        } else {
            self.quantizedWeight = nil
            self.scales = nil
            self.quantBiases = nil
            self.groupSize = nil
            self.bits = nil
            self.mode = .affine
        }
    }

    func matmul(_ x: MLXArray) -> MLXArray {
        if let quantizedWeight, let scales {
            return quantizedMM(
                x,
                quantizedWeight,
                scales: scales,
                biases: quantBiases,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: mode
            )
        }
        return MLX.matmul(x, weight.T)
    }
}

private func linear(_ x: MLXArray, _ weight: CohereLinearWeight, _ bias: MLXArray) -> MLXArray {
    weight.matmul(x) + bias
}

private func linearNoBias(_ x: MLXArray, _ weight: CohereLinearWeight) -> MLXArray {
    weight.matmul(x)
}

private func layerNorm(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray) -> MLXArray {
    MLXFast.layerNorm(x, weight: weight, bias: bias, eps: 1e-5)
}

private func batchNormEval(
    _ x: MLXArray,
    weight: MLXArray,
    bias: MLXArray,
    runningMean: MLXArray,
    runningVar: MLXArray
) -> MLXArray {
    let invStd = 1.0 / sqrt(runningVar + 1e-5)
    let normalized = (x - runningMean.reshaped([1, 1, runningMean.shape[0]]))
        * invStd.reshaped([1, 1, invStd.shape[0]])
    return normalized * weight.reshaped([1, 1, weight.shape[0]])
        + bias.reshaped([1, 1, bias.shape[0]])
}

private func conv2dNHWC(
    _ x: MLXArray,
    weight: MLXArray,
    bias: MLXArray,
    stride: Int,
    padding: Int,
    groups: Int = 1
) -> MLXArray {
    let w = weight.transposed(0, 2, 3, 1)
    return conv2d(x, w, stride: .init(stride), padding: .init(padding), groups: groups)
        + bias.reshaped([1, 1, 1, bias.shape[0]])
}

private func conv1dNLC(
    _ x: MLXArray,
    weight: MLXArray,
    bias: MLXArray,
    stride: Int = 1,
    padding: Int = 0,
    groups: Int = 1
) -> MLXArray {
    let w = weight.transposed(0, 2, 1)
    return conv1d(x, w, stride: stride, padding: padding, groups: groups)
        + bias.reshaped([1, 1, bias.shape[0]])
}

private final class CohereConvSubsampling {
    private let c0w: MLXArray
    private let c0b: MLXArray
    private let c2w: MLXArray
    private let c2b: MLXArray
    private let c3w: MLXArray
    private let c3b: MLXArray
    private let c5w: MLXArray
    private let c5b: MLXArray
    private let c6w: MLXArray
    private let c6b: MLXArray
    private let outW: CohereLinearWeight
    private let outB: MLXArray
    private let convChannels = 256

    init(weights: CohereWeightStore, prefix: String) throws {
        self.c0w = try weights.tensor("\(prefix)conv.0.weight")
        self.c0b = try weights.tensor("\(prefix)conv.0.bias")
        self.c2w = try weights.tensor("\(prefix)conv.2.weight")
        self.c2b = try weights.tensor("\(prefix)conv.2.bias")
        self.c3w = try weights.tensor("\(prefix)conv.3.weight")
        self.c3b = try weights.tensor("\(prefix)conv.3.bias")
        self.c5w = try weights.tensor("\(prefix)conv.5.weight")
        self.c5b = try weights.tensor("\(prefix)conv.5.bias")
        self.c6w = try weights.tensor("\(prefix)conv.6.weight")
        self.c6b = try weights.tensor("\(prefix)conv.6.bias")
        self.outW = try weights.linear("\(prefix)out.weight")
        self.outB = try weights.tensor("\(prefix)out.bias")
    }

    func forward(_ input: MLXArray) -> (MLXArray, Int) {
        var x = input.transposed(0, 2, 1).expandedDimensions(axis: -1)
        x = relu(conv2dNHWC(x, weight: c0w, bias: c0b, stride: 2, padding: 1))
        x = conv2dNHWC(
            x, weight: c2w, bias: c2b, stride: 2, padding: 1, groups: convChannels)
        x = relu(conv2dNHWC(x, weight: c3w, bias: c3b, stride: 1, padding: 0))
        x = conv2dNHWC(
            x, weight: c5w, bias: c5b, stride: 2, padding: 1, groups: convChannels)
        x = relu(conv2dNHWC(x, weight: c6w, bias: c6b, stride: 1, padding: 0))

        let shape = x.shape
        let batch = shape[0]
        let steps = shape[1]
        x = x.transposed(0, 1, 3, 2).reshaped([batch, steps, -1])
        return (linear(x, outW, outB), steps)
    }
}

private func relPositionalEncoding(length: Int, dModel: Int, dtype: DType) -> MLXArray {
    let nPos = max((2 * length) - 1, 1)
    var values = Array(repeating: Float(0), count: nPos * dModel)
    for i in 0 ..< nPos {
        let pos = Double(length - 1 - i)
        var k = 0
        while k < dModel {
            let div = exp(Double(k) * -log(10000.0) / Double(dModel))
            values[(i * dModel) + k] = Float(sin(pos * div))
            if k + 1 < dModel {
                values[(i * dModel) + k + 1] = Float(cos(pos * div))
            }
            k += 2
        }
    }
    return MLXArray(values, [1, nPos, dModel]).asType(dtype)
}

private final class CohereFeedForward {
    private let l1w: CohereLinearWeight
    private let l1b: MLXArray
    private let l2w: CohereLinearWeight
    private let l2b: MLXArray

    init(weights: CohereWeightStore, prefix: String) throws {
        self.l1w = try weights.linear("\(prefix)linear1.weight")
        self.l1b = try weights.tensor("\(prefix)linear1.bias")
        self.l2w = try weights.linear("\(prefix)linear2.weight")
        self.l2b = try weights.tensor("\(prefix)linear2.bias")
    }

    func forward(_ input: MLXArray) -> MLXArray {
        linear(silu(linear(input, l1w, l1b)), l2w, l2b)
    }
}

private final class CohereConformerConv {
    private let pw1w: MLXArray
    private let pw1b: MLXArray
    private let dww: MLXArray
    private let dwb: MLXArray
    private let bnw: MLXArray
    private let bnb: MLXArray
    private let bnMean: MLXArray
    private let bnVar: MLXArray
    private let pw2w: MLXArray
    private let pw2b: MLXArray
    private let dModel: Int

    init(weights: CohereWeightStore, prefix: String, dModel: Int) throws {
        self.pw1w = try weights.tensor("\(prefix)pointwise_conv1.weight")
        self.pw1b = try weights.tensor("\(prefix)pointwise_conv1.bias")
        self.dww = try weights.tensor("\(prefix)depthwise_conv.weight")
        self.dwb = try weights.tensor("\(prefix)depthwise_conv.bias")
        self.bnw = try weights.tensor("\(prefix)batch_norm.weight")
        self.bnb = try weights.tensor("\(prefix)batch_norm.bias")
        self.bnMean = try weights.tensor("\(prefix)batch_norm.running_mean")
        self.bnVar = try weights.tensor("\(prefix)batch_norm.running_var")
        self.pw2w = try weights.tensor("\(prefix)pointwise_conv2.weight")
        self.pw2b = try weights.tensor("\(prefix)pointwise_conv2.bias")
        self.dModel = dModel
    }

    func forward(_ input: MLXArray) -> MLXArray {
        var x = conv1dNLC(input, weight: pw1w, bias: pw1b)
        let parts = x.split(parts: 2, axis: -1)
        x = parts[0] * sigmoid(parts[1])
        let kernel = dww.shape[2]
        x = conv1dNLC(x, weight: dww, bias: dwb, padding: (kernel - 1) / 2, groups: dModel)
        x = batchNormEval(x, weight: bnw, bias: bnb, runningMean: bnMean, runningVar: bnVar)
        x = silu(x)
        return conv1dNLC(x, weight: pw2w, bias: pw2b)
    }
}

private final class CohereRelPosAttention {
    private let qw: CohereLinearWeight
    private let qb: MLXArray
    private let kw: CohereLinearWeight
    private let kb: MLXArray
    private let vw: CohereLinearWeight
    private let vb: MLXArray
    private let posW: CohereLinearWeight
    private let outW: CohereLinearWeight
    private let outB: MLXArray
    private let posBiasU: MLXArray
    private let posBiasV: MLXArray
    private let nHeads: Int
    private let headDim: Int
    private let scale: Float

    init(weights: CohereWeightStore, prefix: String, nHeads: Int, dModel: Int) throws {
        self.qw = try weights.linear("\(prefix)linear_q.weight")
        self.qb = try weights.tensor("\(prefix)linear_q.bias")
        self.kw = try weights.linear("\(prefix)linear_k.weight")
        self.kb = try weights.tensor("\(prefix)linear_k.bias")
        self.vw = try weights.linear("\(prefix)linear_v.weight")
        self.vb = try weights.tensor("\(prefix)linear_v.bias")
        self.posW = try weights.linear("\(prefix)linear_pos.weight")
        self.outW = try weights.linear("\(prefix)linear_out.weight")
        self.outB = try weights.tensor("\(prefix)linear_out.bias")
        self.posBiasU = try weights.tensor("\(prefix)pos_bias_u")
        self.posBiasV = try weights.tensor("\(prefix)pos_bias_v")
        self.nHeads = nHeads
        self.headDim = dModel / nHeads
        self.scale = Float(1.0 / sqrt(Double(self.headDim)))
    }

    func forward(_ input: MLXArray, posEmb: MLXArray, debugDumpPrefix: String? = nil) throws
        -> MLXArray
    {
        func dumpStage(_ name: String, _ array: MLXArray) throws {
            guard let debugDumpPrefix, !debugDumpPrefix.isEmpty else {
                return
            }
            try dumpArray(array, path: "\(debugDumpPrefix)-att_\(name).f32le", label: "layer0.att.\(name)")
        }

        let shape = input.shape
        let batch = shape[0]
        let steps = shape[1]
        let qLinear = linear(input, qw, qb)
        let kLinear = linear(input, kw, kb)
        let vLinear = linear(input, vw, vb)
        try dumpStage("q_linear", qLinear)
        try dumpStage("k_linear", kLinear)
        try dumpStage("v_linear", vLinear)
        let q = reshapeHeads(qLinear, batch: batch, steps: steps)
        let k = reshapeHeads(kLinear, batch: batch, steps: steps)
        let v = reshapeHeads(vLinear, batch: batch, steps: steps)
        try dumpStage("q", q)
        try dumpStage("k", k)
        try dumpStage("v", v)

        let nPos = posEmb.shape[1]
        let p = linearNoBias(posEmb, posW)
            .reshaped([1, nPos, nHeads, headDim])
            .transposed(0, 2, 1, 3)
        try dumpStage("p", p)

        let u = posBiasU.reshaped([1, nHeads, 1, headDim])
        let vBias = posBiasV.reshaped([1, nHeads, 1, headDim])
        let qWithU = q + u
        let qWithV = q + vBias
        try dumpStage("q_u", qWithU)
        try dumpStage("q_v", qWithV)
        let matrixAC = matmul(qWithU, k.transposed(0, 1, 3, 2))
        let matrixBDRaw = matmul(qWithV, p.transposed(0, 1, 3, 2))
        let matrixBD = relShift(matrixBDRaw)
        try dumpStage("matrix_ac", matrixAC)
        try dumpStage("matrix_bd_raw", matrixBDRaw)
        try dumpStage("matrix_bd", matrixBD)
        let scores = (matrixAC + matrixBD) * scale
        try dumpStage("scores", scores)
        let probs = softmax(scores, axis: -1)
        try dumpStage("probs", probs)
        let attended = matmul(probs, v)
        try dumpStage("attended", attended)
        let out = attended.transposed(0, 2, 1, 3).reshaped([batch, steps, nHeads * headDim])
        try dumpStage("merged", out)
        let projected = linear(out, outW, outB)
        try dumpStage("projected", projected)
        return projected
    }

    private func reshapeHeads(_ x: MLXArray, batch: Int, steps: Int) -> MLXArray {
        x.reshaped([batch, steps, nHeads, headDim]).transposed(0, 2, 1, 3)
    }

    private func relShift(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let batch = shape[0]
        let heads = shape[1]
        let steps = shape[2]
        let paddedX = padded(
            x,
            widths: [
                IntOrPair(0),
                IntOrPair(0),
                IntOrPair(0),
                IntOrPair((1, 0)),
            ]
        )
        let reshaped = paddedX.reshaped([batch, heads, -1, steps])
        let shifted = reshaped[0..., 0..., 1 ..< (2 * steps), 0...]
            .reshaped([batch, heads, steps, (2 * steps) - 1])
        return shifted[0..., 0..., 0..., 0 ..< steps]
    }
}

private final class CohereConformerLayer {
    private let normFF1W: MLXArray
    private let normFF1B: MLXArray
    private let ff1: CohereFeedForward
    private let normAttW: MLXArray
    private let normAttB: MLXArray
    private let attention: CohereRelPosAttention
    private let normConvW: MLXArray
    private let normConvB: MLXArray
    private let conv: CohereConformerConv
    private let normFF2W: MLXArray
    private let normFF2B: MLXArray
    private let ff2: CohereFeedForward
    private let normOutW: MLXArray
    private let normOutB: MLXArray

    init(weights: CohereWeightStore, prefix: String, nHeads: Int, dModel: Int) throws {
        self.normFF1W = try weights.tensor("\(prefix)norm_feed_forward1.weight")
        self.normFF1B = try weights.tensor("\(prefix)norm_feed_forward1.bias")
        self.ff1 = try CohereFeedForward(weights: weights, prefix: "\(prefix)feed_forward1.")
        self.normAttW = try weights.tensor("\(prefix)norm_self_att.weight")
        self.normAttB = try weights.tensor("\(prefix)norm_self_att.bias")
        self.attention = try CohereRelPosAttention(
            weights: weights, prefix: "\(prefix)self_attn.", nHeads: nHeads, dModel: dModel)
        self.normConvW = try weights.tensor("\(prefix)norm_conv.weight")
        self.normConvB = try weights.tensor("\(prefix)norm_conv.bias")
        self.conv = try CohereConformerConv(
            weights: weights, prefix: "\(prefix)conv.", dModel: dModel)
        self.normFF2W = try weights.tensor("\(prefix)norm_feed_forward2.weight")
        self.normFF2B = try weights.tensor("\(prefix)norm_feed_forward2.bias")
        self.ff2 = try CohereFeedForward(weights: weights, prefix: "\(prefix)feed_forward2.")
        self.normOutW = try weights.tensor("\(prefix)norm_out.weight")
        self.normOutB = try weights.tensor("\(prefix)norm_out.bias")
    }

    func forward(_ input: MLXArray, posEmb: MLXArray, debugDumpPrefix: String? = nil) throws
        -> MLXArray
    {
        func dumpStage(_ name: String, _ array: MLXArray) throws {
            guard let debugDumpPrefix, !debugDumpPrefix.isEmpty else {
                return
            }
            try dumpArray(array, path: "\(debugDumpPrefix)-\(name).f32le", label: "layer0.\(name)")
        }

        try dumpStage("input", input)
        let normFF1 = layerNorm(input, normFF1W, normFF1B)
        try dumpStage("norm_ff1", normFF1)
        let ff1Out = ff1.forward(normFF1)
        try dumpStage("ff1_out", ff1Out)
        var x = input + (ff1Out * 0.5)
        try dumpStage("after_ff1", x)
        let normAtt = layerNorm(x, normAttW, normAttB)
        try dumpStage("norm_att", normAtt)
        let attOut = try attention.forward(
            normAtt,
            posEmb: posEmb,
            debugDumpPrefix: debugDumpPrefix
        )
        try dumpStage("att_out", attOut)
        x = x + attOut
        try dumpStage("after_att", x)
        let normConv = layerNorm(x, normConvW, normConvB)
        try dumpStage("norm_conv", normConv)
        let convOut = conv.forward(normConv)
        try dumpStage("conv_out", convOut)
        x = x + convOut
        try dumpStage("after_conv", x)
        let normFF2 = layerNorm(x, normFF2W, normFF2B)
        try dumpStage("norm_ff2", normFF2)
        let ff2Out = ff2.forward(normFF2)
        try dumpStage("ff2_out", ff2Out)
        x = x + (ff2Out * 0.5)
        try dumpStage("before_norm_out", x)
        let out = layerNorm(x, normOutW, normOutB)
        try dumpStage("out", out)
        return out
    }
}

private final class CohereConformerEncoder {
    private let preEncode: CohereConvSubsampling
    private let layers: [CohereConformerLayer]
    private let projectionW: CohereLinearWeight?
    private let projectionB: MLXArray?
    private let dModel: Int

    init(config: CohereModelConfig, weights: CohereWeightStore) throws {
        guard let encoderConfig = config.encoder else {
            throw ASRMLXRuntimeError.invalidBundle("Cohere config is missing encoder settings")
        }
        self.dModel = encoderConfig.dModel
        self.preEncode = try CohereConvSubsampling(weights: weights, prefix: "encoder.pre_encode.")
        var layers = [CohereConformerLayer]()
        layers.reserveCapacity(encoderConfig.nLayers)
        for index in 0 ..< encoderConfig.nLayers {
            layers.append(
                try CohereConformerLayer(
                    weights: weights,
                    prefix: "encoder.layers.\(index).",
                    nHeads: encoderConfig.nHeads,
                    dModel: encoderConfig.dModel
                )
            )
        }
        self.layers = layers
        self.projectionW = try? weights.linear("encoder_decoder_proj.weight")
        self.projectionB = try? weights.tensor("encoder_decoder_proj.bias")
    }

    func forward(_ features: MLXArray) throws -> MLXArray {
        let (subsampled, steps) = preEncode.forward(features)
        try dumpArray(subsampled, envName: "ASR_COHERE_MLX_DUMP_PREENCODE")
        let posEmb = relPositionalEncoding(length: steps, dModel: dModel, dtype: subsampled.dtype)
        var x = subsampled
        let layer0DumpPrefix = ProcessInfo.processInfo.environment["ASR_COHERE_MLX_DUMP_LAYER0_PREFIX"]
        for (index, layer) in layers.enumerated() {
            x = try layer.forward(
                x,
                posEmb: posEmb,
                debugDumpPrefix: index == 0 ? layer0DumpPrefix : nil
            )
            if index == 0 {
                try dumpArray(x, envName: "ASR_COHERE_MLX_DUMP_LAYER0")
            }
        }
        if let projectionW, let projectionB {
            x = linear(x, projectionW, projectionB)
        }
        return x
    }
}

private struct CohereSelfKV {
    var key: MLXArray?
    var value: MLXArray?
}

private struct CohereCrossKV {
    let key: MLXArray
    let value: MLXArray
}

private final class CohereDecoderAttention {
    let qw: CohereLinearWeight
    let qb: MLXArray
    let kw: CohereLinearWeight
    let kb: MLXArray
    let vw: CohereLinearWeight
    let vb: MLXArray
    let outW: CohereLinearWeight
    let outB: MLXArray
    let nHeads: Int
    let headDim: Int
    let hidden: Int

    init(weights: CohereWeightStore, prefix: String, nHeads: Int, hidden: Int) throws {
        self.qw = try weights.linear("\(prefix)query_net.weight")
        self.qb = try weights.tensor("\(prefix)query_net.bias")
        self.kw = try weights.linear("\(prefix)key_net.weight")
        self.kb = try weights.tensor("\(prefix)key_net.bias")
        self.vw = try weights.linear("\(prefix)value_net.weight")
        self.vb = try weights.tensor("\(prefix)value_net.bias")
        self.outW = try weights.linear("\(prefix)out_projection.weight")
        self.outB = try weights.tensor("\(prefix)out_projection.bias")
        self.nHeads = nHeads
        self.headDim = hidden / nHeads
        self.hidden = hidden
    }

    func projectQKV(hiddenStates: MLXArray, source: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let batch = hiddenStates.shape[0]
        let targetSteps = hiddenStates.shape[1]
        let sourceSteps = source.shape[1]
        let q = linear(hiddenStates, qw, qb)
            .reshaped([batch, targetSteps, nHeads, headDim])
            .transposed(0, 2, 1, 3)
        let k = linear(source, kw, kb)
            .reshaped([batch, sourceSteps, nHeads, headDim])
            .transposed(0, 2, 1, 3)
        let v = linear(source, vw, vb)
            .reshaped([batch, sourceSteps, nHeads, headDim])
            .transposed(0, 2, 1, 3)
        return (q, k, v)
    }
}

private final class CohereDecoderFFN {
    private let denseInW: CohereLinearWeight
    private let denseInB: MLXArray
    private let denseOutW: CohereLinearWeight
    private let denseOutB: MLXArray

    init(weights: CohereWeightStore, prefix: String) throws {
        self.denseInW = try weights.linear("\(prefix)dense_in.weight")
        self.denseInB = try weights.tensor("\(prefix)dense_in.bias")
        self.denseOutW = try weights.linear("\(prefix)dense_out.weight")
        self.denseOutB = try weights.tensor("\(prefix)dense_out.bias")
    }

    func forward(_ input: MLXArray) -> MLXArray {
        linear(relu(linear(input, denseInW, denseInB)), denseOutW, denseOutB)
    }
}

private final class CohereDecoderLayer {
    private let norm1W: MLXArray
    private let norm1B: MLXArray
    private let selfAttention: CohereDecoderAttention
    private let norm2W: MLXArray
    private let norm2B: MLXArray
    let crossAttention: CohereDecoderAttention
    private let norm3W: MLXArray
    private let norm3B: MLXArray
    private let ffn: CohereDecoderFFN

    init(weights: CohereWeightStore, prefix: String, nHeads: Int, hidden: Int) throws {
        self.norm1W = try weights.tensor("\(prefix)layer_norm_1.weight")
        self.norm1B = try weights.tensor("\(prefix)layer_norm_1.bias")
        self.selfAttention = try CohereDecoderAttention(
            weights: weights, prefix: "\(prefix)first_sub_layer.", nHeads: nHeads, hidden: hidden)
        self.norm2W = try weights.tensor("\(prefix)layer_norm_2.weight")
        self.norm2B = try weights.tensor("\(prefix)layer_norm_2.bias")
        self.crossAttention = try CohereDecoderAttention(
            weights: weights, prefix: "\(prefix)second_sub_layer.", nHeads: nHeads, hidden: hidden)
        self.norm3W = try weights.tensor("\(prefix)layer_norm_3.weight")
        self.norm3B = try weights.tensor("\(prefix)layer_norm_3.bias")
        self.ffn = try CohereDecoderFFN(weights: weights, prefix: "\(prefix)third_sub_layer.")
    }

    func forwardCached(
        hidden: MLXArray,
        selfKV: CohereSelfKV,
        crossKV: CohereCrossKV
    ) -> (MLXArray, CohereSelfKV) {
        let normalized = layerNorm(hidden, norm1W, norm1B)
        let batch = normalized.shape[0]
        let steps = normalized.shape[1]
        let (qNew, kNew, vNew) = selfAttention.projectQKV(
            hiddenStates: normalized, source: normalized)

        let kFull: MLXArray
        let vFull: MLXArray
        if let key = selfKV.key, let value = selfKV.value {
            kFull = concatenated([key, kNew], axis: 2)
            vFull = concatenated([value, vNew], axis: 2)
        } else {
            kFull = kNew
            vFull = vNew
        }

        var selfOut = scaledDotProductAttention(q: qNew, k: kFull, v: vFull)
        selfOut = selfOut.transposed(0, 2, 1, 3)
            .reshaped([batch, steps, selfAttention.hidden])
        var nextHidden = hidden + linear(selfOut, selfAttention.outW, selfAttention.outB)

        let normalized2 = layerNorm(nextHidden, norm2W, norm2B)
        let crossQ = linear(normalized2, crossAttention.qw, crossAttention.qb)
            .reshaped([batch, steps, crossAttention.nHeads, crossAttention.headDim])
            .transposed(0, 2, 1, 3)
        var crossOut = scaledDotProductAttention(q: crossQ, k: crossKV.key, v: crossKV.value)
        crossOut = crossOut.transposed(0, 2, 1, 3)
            .reshaped([batch, steps, crossAttention.hidden])
        nextHidden = nextHidden + linear(crossOut, crossAttention.outW, crossAttention.outB)

        let normalized3 = layerNorm(nextHidden, norm3W, norm3B)
        nextHidden = nextHidden + ffn.forward(normalized3)
        return (nextHidden, CohereSelfKV(key: kFull, value: vFull))
    }
}

private func scaledDotProductAttention(q: MLXArray, k: MLXArray, v: MLXArray) -> MLXArray {
    let scale = Float(1.0 / sqrt(Double(q.shape.last ?? 1)))
    let scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
    return matmul(softmax(scores, axis: -1), v)
}

private final class CohereTransformerDecoder {
    private let tokenEmbedding: MLXArray
    private let posEmbedding: MLXArray
    private let embNormW: MLXArray
    private let embNormB: MLXArray
    private let layers: [CohereDecoderLayer]
    private let finalNormW: MLXArray
    private let finalNormB: MLXArray
    private let headW: CohereLinearWeight
    private let headB: MLXArray
    private let nHeads: Int
    private let headDim: Int

    var layerCount: Int {
        layers.count
    }

    init(config: CohereModelConfig, weights: CohereWeightStore) throws {
        guard let decoderConfig = config.transfDecoder?.configDict else {
            throw ASRMLXRuntimeError.invalidBundle("Cohere config is missing decoder settings")
        }
        let hidden = decoderConfig.hiddenSize
        self.nHeads = decoderConfig.numAttentionHeads
        self.headDim = hidden / decoderConfig.numAttentionHeads
        self.tokenEmbedding = try weights.tensor("transf_decoder._embedding.token_embedding.weight")
        self.posEmbedding = try weights.tensor("transf_decoder._embedding.position_embedding.pos_enc")
        self.embNormW = try weights.tensor("transf_decoder._embedding.layer_norm.weight")
        self.embNormB = try weights.tensor("transf_decoder._embedding.layer_norm.bias")

        var layers = [CohereDecoderLayer]()
        layers.reserveCapacity(decoderConfig.numLayers)
        for index in 0 ..< decoderConfig.numLayers {
            layers.append(
                try CohereDecoderLayer(
                    weights: weights,
                    prefix: "transf_decoder._decoder.layers.\(index).",
                    nHeads: decoderConfig.numAttentionHeads,
                    hidden: hidden
                )
            )
        }
        self.layers = layers
        self.finalNormW = try weights.tensor("transf_decoder._decoder.final_layer_norm.weight")
        self.finalNormB = try weights.tensor("transf_decoder._decoder.final_layer_norm.bias")
        self.headW = try weights.linear("log_softmax.mlp.layer0.weight")
        self.headB = try weights.tensor("log_softmax.mlp.layer0.bias")
    }

    func precomputeCrossKV(_ encoderHidden: MLXArray) throws -> [CohereCrossKV] {
        let batch = encoderHidden.shape[0]
        let sourceSteps = encoderHidden.shape[1]
        return layers.map { layer in
            let attn = layer.crossAttention
            let k = linear(encoderHidden, attn.kw, attn.kb)
                .reshaped([batch, sourceSteps, nHeads, headDim])
                .transposed(0, 2, 1, 3)
            let v = linear(encoderHidden, attn.vw, attn.vb)
                .reshaped([batch, sourceSteps, nHeads, headDim])
                .transposed(0, 2, 1, 3)
            return CohereCrossKV(key: k, value: v)
        }
    }

    func step(
        tokenID: Int,
        position: Int,
        selfKV: [CohereSelfKV],
        crossKV: [CohereCrossKV]
    ) throws -> (logits: MLXArray, selfKV: [CohereSelfKV]) {
        let ids = MLXArray([Int32(tokenID)])
        let pos = MLXArray([Int32(position)])
        let token = tokenEmbedding.take(ids, axis: 0).expandedDimensions(axis: 0)
        let positionEmbedding = posEmbedding.take(pos, axis: 0).expandedDimensions(axis: 0)
        var hidden = layerNorm(token + positionEmbedding, embNormW, embNormB)
        var nextKV = [CohereSelfKV]()
        nextKV.reserveCapacity(layers.count)

        for (index, layer) in layers.enumerated() {
            let step = layer.forwardCached(
                hidden: hidden,
                selfKV: selfKV[index],
                crossKV: crossKV[index]
            )
            hidden = step.0
            nextKV.append(step.1)
        }

        hidden = layerNorm(hidden, finalNormW, finalNormB)
        let logits = linear(hidden[0..., 0, 0...], headW, headB)
        return (logits, nextKV)
    }
}

private func argmaxToken(_ logits: MLXArray) -> Int {
    let token = argMax(logits, axis: -1)
    eval(token)
    return Int(token.item(Int32.self))
}
