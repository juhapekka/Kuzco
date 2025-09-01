//
//  LlamaInstance.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation
import llama

@globalActor public actor LlamaInstanceActor {
    public static let shared = LlamaInstanceActor()
}

@LlamaInstanceActor
public class LlamaInstance {
    public let profile: ModelProfile
    internal var settings: InstanceSettings
    private var predictionCfg: PredictionConfig

    private var clModel: CLlamaModel?
    private var clContext: CLlamaContext?
    private var clBatch: CLlamaBatch?

    private var eosHandler: EosHandler
    private var interactionFormatter: InteractionFormatting

    private var currentContextTokens: [CLlamaToken] = []
    // Speak-mode cache: concrete token IDs for each SNAC step 0..6
    private var speakStepIdCache: [[CLlamaToken]] = Array(repeating: [], count: 7)
    private var speakStepCacheModelId: String? = nil
    private var lastAudioTokPerStep: [Int32] = Array(repeating: -1, count: 7)

    private var interruptFlag: Bool = false
    private var loadingProgressContinuation: AsyncStream<LoadUpdate>.Continuation?

    public var isReady: Bool { clModel != nil && clContext != nil && clBatch != nil }
    

    public init(
        profile: ModelProfile,
        settings: InstanceSettings,
        predictionConfig: PredictionConfig,
        formatter: InteractionFormatting? = nil,
        customStopSequences: [String] = []
    ) {
        self.profile = profile
        self.settings = settings
        self.predictionCfg = predictionConfig
        self.interactionFormatter = formatter ?? StandardInteractionFormatter()
        self.eosHandler = EosHandler(architecture: profile.architecture, additionalStopSequences: customStopSequences)
        LlamaKitBridge.initializeLlamaBackend()
    }

    deinit {
        let modelToFree = self.clModel
        let contextToFree = self.clContext
        let batchToFree = self.clBatch
        let profileIDForLog = self.profile.id

        self.loadingProgressContinuation?.finish()

        if modelToFree != nil || contextToFree != nil || batchToFree != nil {
            Task.detached { @LlamaInstanceActor in
                if let batch = batchToFree {
                    LlamaKitBridge.freeBatch(batch)
                    print("Kuzco LlamaInstance (\(profileIDForLog)) C batch freed via deinit task.")
                }
                if let context = contextToFree {
                    LlamaKitBridge.freeContext(context)
                    print("Kuzco LlamaInstance (\(profileIDForLog)) C context freed via deinit task.")
                }
                if let model = modelToFree {
                    LlamaKitBridge.freeModel(model)
                    print("Kuzco LlamaInstance (\(profileIDForLog)) C model freed via deinit task.")
                }
            }
        }
        print("Kuzco LlamaInstance for \(profileIDForLog) deinit sequence initiated.")
    }


    private func publishProgress(_ update: LoadUpdate) {
        loadingProgressContinuation?.yield(update)
        if update.stage == .ready || update.stage == .failed {
            loadingProgressContinuation?.finish()
            loadingProgressContinuation = nil
        }
    }

    public func startup() -> AsyncStream<LoadUpdate> {
        return AsyncStream { continuation in
            self.loadingProgressContinuation = continuation
            Task { @LlamaInstanceActor [] in
                await performStartup()
            }
        }
    }

    private func performStartup() async {
        guard !isReady else {
            publishProgress(LoadUpdate(stage: .ready, detail: "Instance already initialized."))
            return
        }
        publishProgress(LoadUpdate(stage: .preparing))

        do {
            // Validate model file before attempting to load
            publishProgress(LoadUpdate(stage: .readingModel, detail: "Validating model file at \(profile.sourcePath)"))
            
            // Use the bridge's validation function
            try LlamaKitBridge.validateModelFile(path: profile.sourcePath)
            publishProgress(LoadUpdate(stage: .readingModel, detail: "Model file validation passed"))

            // Attempt to load the model with enhanced error handling
            publishProgress(LoadUpdate(stage: .readingModel, detail: "Loading model data..."))
            
            do {
                self.clModel = try LlamaKitBridge.loadModel(from: profile.sourcePath, settings: self.settings)
                publishProgress(LoadUpdate(stage: .readingModel, detail: "Model data loaded successfully."))
            } catch let error as KuzcoError {
                // Handle specific model loading errors
                switch error {
                case .unsupportedModelArchitecture(let architecture, let suggestion):
                    publishProgress(LoadUpdate(
                        stage: .failed,
                        detail: "Unsupported model architecture '\(architecture)': \(suggestion)",
                        hasError: true
                    ))
                    await performShutdownInternal()
                    return
                case .modelInitializationFailed(let details):
                    publishProgress(LoadUpdate(
                        stage: .failed,
                        detail: "Model loading failed: \(details)",
                        hasError: true
                    ))
                    await performShutdownInternal()
                    return
                default:
                    throw error
                }
            }

            publishProgress(LoadUpdate(stage: .creatingContext, detail: "Initializing inference context..."))
            guard let model = self.clModel else { 
                throw KuzcoError.modelInitializationFailed(details: "Model pointer nil after successful load - this is unexpected") 
            }
            
            // Create context with error handling
            do {
                self.clContext = try LlamaKitBridge.createContext(for: model, settings: self.settings)
            } catch let error as KuzcoError {
                switch error {
                case .contextCreationFailed(let details):
                    // Try with reduced settings as fallback
                    publishProgress(LoadUpdate(stage: .creatingContext, detail: "Primary context creation failed, trying fallback settings..."))
                    
                    var fallbackSettings = self.settings
                    fallbackSettings.contextLength = min(self.settings.contextLength, 2048)
                    fallbackSettings.processingBatchSize = min(self.settings.processingBatchSize, 256)
                    fallbackSettings.offloadedGpuLayers = 0 // CPU only
                    
                    do {
                        self.clContext = try LlamaKitBridge.createContext(for: model, settings: fallbackSettings)
                        self.settings = fallbackSettings // Update to working settings
                        publishProgress(LoadUpdate(stage: .creatingContext, detail: "Context created with fallback settings (CPU-only, reduced context)"))
                    } catch {
                        throw KuzcoError.contextCreationFailed(details: "Both primary and fallback context creation failed: \(details)")
                    }
                default:
                    throw error
                }
            }

            // Validate context
            if let ctx = self.clContext {
                let modelMaxCtx = LlamaKitBridge.getModelMaxContextLength(context: ctx)
                if self.settings.contextLength > modelMaxCtx {
                     print("Kuzco Notice: Requested context \(self.settings.contextLength) > model max \(modelMaxCtx). Using model max for instance setting.")
                    self.settings.contextLength = modelMaxCtx
                }
            }
            publishProgress(LoadUpdate(stage: .creatingContext, detail: "Context validated."))

            // Create batch with error handling
            do {
                self.clBatch = try LlamaKitBridge.createBatch(maxTokens: self.settings.processingBatchSize)
                publishProgress(LoadUpdate(stage: .creatingContext, detail: "Token batch created."))
            } catch {
                throw KuzcoError.batchCreationFailed
            }

            // Pre-warm with extra safety
            publishProgress(LoadUpdate(stage: .prewarming, detail: "Pre-warming inference engine..."))
            do {
                try await prewarmEngine()
                publishProgress(LoadUpdate(stage: .prewarming, detail: "Engine pre-warmed successfully."))
            } catch let error as KuzcoError {
                // Pre-warming failure is often non-fatal, continue with warning
                print("🦙 Kuzco Warning: Pre-warming failed: \(error.localizedDescription). Continuing anyway. 🦙")
                publishProgress(LoadUpdate(stage: .prewarming, detail: "Pre-warming failed but continuing (non-fatal)."))
            }

            publishProgress(LoadUpdate(stage: .ready, detail: "Model ready for inference"))
            
        } catch let error as KuzcoError {
            print("🦙 Kuzco Startup Error: \(error.localizedDescription) 🦙")
            await performShutdownInternal()
            publishProgress(LoadUpdate(stage: .failed, detail: error.localizedDescription, hasError: true))
        } catch {
            print("🦙 Kuzco Unexpected Startup Error: \(error.localizedDescription) 🦙")
            await performShutdownInternal()
            publishProgress(LoadUpdate(stage: .failed, detail: "Unexpected error: \(error.localizedDescription)", hasError: true))
        }
    }

    private func prewarmEngine() async throws {
        guard let model = clModel, let context = clContext, self.clBatch != nil else {
            throw KuzcoError.engineNotReady
        }
        do {
            let warmUpPrompt = " "
            let tokens = try LlamaKitBridge.tokenize(text: warmUpPrompt, model: model, addBos: true, parseSpecial: true)
            guard !tokens.isEmpty else { throw KuzcoError.warmUpRoutineFailed(details: "Warm-up tokenization resulted in no tokens.") }

            LlamaKitBridge.clearBatch(&self.clBatch!)
            LlamaKitBridge.addTokenToBatch(batch: &self.clBatch!, token: tokens[0], position: 0, sequenceId: 0, enableLogits: true)

            LlamaKitBridge.setThreads(for: context, mainThreads: 1, batchThreads: 1)
            try LlamaKitBridge.processBatch(context: context, batch: self.clBatch!)
            LlamaKitBridge.clearKeyValueCache(context: context)
            currentContextTokens.removeAll()
        } catch {
            throw KuzcoError.warmUpRoutineFailed(details: "Exception during prewarm: \(error.localizedDescription)")
        }

    }

// Build whitelist for speak-mode: allow only <custom_token_N> and end aliases.
// Prefer <|eot_id|> if present; otherwise take the first alias we find.
private func buildCustomWhitelist(model: CLlamaModel) -> (allowed: Set<Int32>, eot: Int32?) {
    let nVocab = Int(LlamaKitBridge.getVocabSize(model: model))
    var allowed = Set<Int32>()
    allowed.reserveCapacity(8192)
    var eotTok: Int32? = nil

    let endAliases: Set<String> = ["<|eot_id|>", "<|eos|>", "<|endoftext|>", "<|im_end|>"]

    for tid in 0..<nVocab {
        let s = LlamaKitBridge.detokenize(token: Int32(tid), model: model)

        if endAliases.contains(s) {
            // Prefer canonical <|eot_id|>, otherwise take the first alias we encounter.
            if eotTok == nil || s == "<|eot_id|>" {
                eotTok = Int32(tid)
            }
            allowed.insert(Int32(tid))
            continue
        }

        if s.hasPrefix("<custom_token_") && s.hasSuffix(">") {
            let inner = s.dropFirst("<custom_token_".count).dropLast()
            if !inner.isEmpty, Int(inner) != nil {
                allowed.insert(Int32(tid))
            }
        }
    }
    return (allowed, eotTok)

}

    // Precompute, for each SNAC step (0..6), the concrete token IDs for <custom_token_(base + step*vocab + k)>
    // Uses tokenization to resolve the IDs reliably even when detokenize() hides specials.
    private func buildSpeakStepIdCache(model: CLlamaModel, base: Int = 10, vocab: Int = 4096) -> [[CLlamaToken]] {
        var cache = Array(repeating: [CLlamaToken](), count: 7)
        for step in 0..<7 {
            let minBigN = base + step * vocab
            var arr: [CLlamaToken] = []
            arr.reserveCapacity(vocab)
            for k in 0..<vocab {
                let bigN = minBigN + k
                if let tok = self.tokenIdForCustomBigN(bigN) {
                    arr.append(tok)
                }
            }
            cache[step] = arr
        }
        return cache
    }

    // Build a *boost* set (no hard mask): audio codes + helpful specials
    private func buildSpeakBoostSet(model: CLlamaModel) -> (boost: Set<Int32>, eot: Int32?) {
        let nVocab = Int(LlamaKitBridge.getVocabSize(model: model))
        var boost = Set<Int32>(); boost.reserveCapacity(32768)
        var eotTok: Int32? = nil

        let endAliases: Set<String> = ["<|eot_id|>", "<|eos|>", "<|endoftext|>", "<|im_end|>"]
        let expressive: Set<String> = ["<laugh>", "<sigh>", "<groan>", "<chuckle>", "<yawn>", "<gasp>", "<cough>", "<sniffle>"]

        for tid in 0..<nVocab {
            let s = LlamaKitBridge.detokenize(token: Int32(tid), model: model)
            if endAliases.contains(s) {
                eotTok = Int32(tid)
                boost.insert(Int32(tid))
                continue
            }
            if s.hasPrefix("<custom_token_") && s.hasSuffix(">") {
                boost.insert(Int32(tid));
                continue
            }
            if expressive.contains(s) {
                boost.insert(Int32(tid));
                continue
            }
            // Optional: lightly include any other bracketed tags
            if s.hasPrefix("<") && s.hasSuffix(">") {
                boost.insert(Int32(tid));
                continue
            }
        }
        return (boost, eotTok)
    }

    // Map `<custom_token_N>` piece to SNAC id given how many valid codes have been accepted so far.
    // Mirrors Python: id = N - base - ((accepted % 7) * vocab); accept iff 0 <= id < vocab.
    // This parser is tolerant to trailing characters (e.g. whitespace, hidden markers) after N.
    private func snacId(fromCustomPiece piece: String, acceptedSoFar: Int, base: Int = 10, vocab: Int = 4096) -> Int? {
        let prefix = "<custom_token_"
        guard let p0 = piece.range(of: prefix)?.upperBound else { return nil }
        // Scan forward for a run of digits; stop at first non-digit (which may be '>' or something else).
        var p = p0
        var hasDigit = false
        var value: Int = 0
        while p < piece.endIndex {
            let ch = piece[p]
            if ch >= "0" && ch <= "9" {
                hasDigit = true
                // value = value * 10 + (ch - "0")
                if let d = ch.wholeNumberValue {
                    value = value * 10 + d
                }
                p = piece.index(after: p)
                continue
            }
            break
        }
        guard hasDigit else { return nil }
        let bigN = value
        let step = acceptedSoFar % 7
        let id = bigN - base - (step * vocab)
        return (id >= 0 && id < vocab) ? id : nil
    }

    // Resolve a token id for an explicit <custom_token_N> piece (no BOS, parse specials)
    private func tokenIdForCustomBigN(_ bigN: Int) -> CLlamaToken? {
        guard let model = self.clModel else { return nil }
        let piece = "<custom_token_\(bigN)>"
        if let toks = try? LlamaKitBridge.tokenize(text: piece, model: model, addBos: false, parseSpecial: true),
           let t = toks.first {
            return CLlamaToken(t)
        }
        return nil
    }

    // Masked greedy for speak-mode: allow only custom tokens (+EOT), optionally block EOT,
    // and add a positive boost to audio-token logits to avoid early EOT.
    private func maskedGreedyAudioToken(
        logitsPtr: UnsafePointer<Float>,
        vocabSize: Int,
        whitelist: Set<Int32>,
        eotToken: Int32?,
        blockEOT: Bool,
        boost: Float
    ) -> CLlamaToken {
        var logits = Array(UnsafeBufferPointer(start: logitsPtr, count: vocabSize))
        let originalLogits = logits
        let negInf: Float = -1e30

        // Apply whitelist mask, optional EOT block, and boost for audio tokens
        for i in 0..<vocabSize {
            let tid = Int32(i)
            if let eot = eotToken, tid == eot {
                logits[i] = blockEOT ? negInf : (logits[i] + boost)
            } else if whitelist.contains(tid) {
                logits[i] += boost
            } else {
                logits[i] = negInf
            }
        }

        // Argmax over masked logits
        var best = -1
        var bestVal = -Float.infinity
        for i in 0..<vocabSize {
            let v = logits[i]
            if v > bestVal { bestVal = v; best = i }
        }

        // Robust fallback if everything was masked out
        if best < 0 || !bestVal.isFinite || bestVal <= negInf/2 {
            if let eot = eotToken, !blockEOT {
                print("⚠️ maskedGreedyAudioToken: mask collapsed, yielding EOT")
                return CLlamaToken(eot)
            }
            // Fall back to unmasked argmax so we never return tid=0 by accident
            var b2 = 0
            var v2 = -Float.infinity
            for i in 0..<vocabSize {
                let vv = originalLogits[i]
                if vv > v2 { v2 = vv; b2 = i }
            }
            print("⚠️ maskedGreedyAudioToken: hard fallback to unmasked argmax tid=\(b2)")
            return CLlamaToken(b2)
        }

        return CLlamaToken(best)
    }

    // Select a *valid* custom token for the current SNAC step using a precomputed allowed-ID list.
    // If allowEOT == true and eotToken exists, EOT competes (boosted) and also becomes the fallback.
    private func maskedGreedyValidAudioForStep(
        logitsPtr: UnsafePointer<Float>,
        vocabSize: Int,
        acceptedSoFar: Int,
        allowedIds: [CLlamaToken],
        eotToken: Int32?,
        allowEOT: Bool = false,
        eotBoost: Float = 3.0
    ) -> CLlamaToken {
        var logits = Array(UnsafeBufferPointer(start: logitsPtr, count: vocabSize))
        let original = logits
        let negInf: Float = -1e30

        // Hard mask everything
        for i in 0..<vocabSize { logits[i] = negInf }

        // Unmask only the allowed custom IDs for this step
        if !allowedIds.isEmpty {
            for t in allowedIds {
                let idx = Int(t)
                if idx >= 0 && idx < vocabSize {
                    logits[idx] = original[idx]
                }
            }
        }

        // Handle EOT separately
        if let eot = eotToken {
            let idx = Int(eot)
            if idx >= 0 && idx < vocabSize {
                logits[idx] = allowEOT ? (original[idx] + eotBoost) : negInf
            }
        }

        // Anti-loop: rangaise toistamasta samaa tokenia samalla SNAC-stepillä
        let step = acceptedSoFar % 7
        if step >= 0 && step < 7 {
            let lastTok = self.lastAudioTokPerStep[step]
            if lastTok >= 0 {
                let i = Int(lastTok)
                if i >= 0 && i < vocabSize {
                    logits[i] -= 2.5    // kevyt miinus, katkaisee 7-askeleen kierron
                }
            }
        }

        // Argmax
        var best = -1
        var bestVal = -Float.infinity
        for i in 0..<vocabSize {
            let v = logits[i]
            if v > bestVal { bestVal = v; best = i }
        }

        // Fallbacks
        if best < 0 || !bestVal.isFinite || bestVal <= negInf/2 {
            if allowEOT, let eot = eotToken {
                print("⚠️ maskedGreedyValidAudioForStep: mask collapsed, yielding EOT")
                return CLlamaToken(eot)
            }
            if !allowedIds.isEmpty {
                // choose the highest-logit among allowed IDs from the unmasked distribution
                var b2 = Int(allowedIds[0])
                var v2 = original[b2]
                for t in allowedIds.dropFirst() {
                    let i = Int(t)
                    let vv = original[i]
                    if vv > v2 { v2 = vv; b2 = i }
                }
                print("⚠️ maskedGreedyValidAudioForStep: mask collapsed, forcing best-allowed id=\(b2)")
                if step >= 0 && step < 7 { self.lastAudioTokPerStep[step] = Int32(b2) }
                return CLlamaToken(b2)
            }
            // last resort: global argmax
            var b3 = 0
            var v3 = -Float.infinity
            for i in 0..<vocabSize {
                let vv = original[i]
                if vv > v3 { v3 = vv; b3 = i }
            }
            print("⚠️ maskedGreedyValidAudioForStep: hard fallback to global argmax tid=\(b3)")
            // Jos valittu osuu allowedIds-joukkoon (eli on tämän stepin audio-koodi), päivitä muisti
            var wasAudio = false
            for t in allowedIds { if Int(t) == b3 { wasAudio = true; break } }
            if wasAudio && step >= 0 && step < 7 { self.lastAudioTokPerStep[step] = Int32(b3) }
            return CLlamaToken(b3)
        }

        // Muista valittu audio-token tälle stepille (ei päivitetä jos se oli EOT)
        var choseAudio = true
        if let eot = eotToken, best == Int(eot) { choseAudio = false }
        if choseAudio && step >= 0 && step < 7 { self.lastAudioTokPerStep[step] = Int32(best) }
        return CLlamaToken(best)    
    }

    // Speak-mode sampler with *soft boost* (no hard masking): repetition penalty + temp/top-k/top-p
    private func boostedSampleForSpeak(
        logitsPtr: UnsafePointer<Float>,
        vocabSize: Int,
        cfg: PredictionConfig,
        recentTokens: [CLlamaToken],
        boostSet: Set<Int32>,
        boostAmount: Float = 2.0,
        repeatWindow: Int = 128
    ) -> CLlamaToken {
        var logits = Array(UnsafeBufferPointer(start: logitsPtr, count: vocabSize))

        // repetition penalty
        if cfg.repetitionPenalty > 1.0, !recentTokens.isEmpty {
            let start = max(0, recentTokens.count - repeatWindow)
            for t in recentTokens[start...] {
                let i = Int(t)
                if i >= 0 && i < vocabSize {
                    let v = logits[i]
                    logits[i] = v >= 0 ? (v / Float(cfg.repetitionPenalty)) : (v * Float(cfg.repetitionPenalty))
                }
            }
        }

        // gentle positive bias for preferred tokens (audio codes + specials)
        for i in 0..<vocabSize {
            if boostSet.contains(Int32(i)) { logits[i] += boostAmount }
        }

        // temperature
        let temperature = max(0.05, cfg.temperature)
        if temperature != 1.0 {
            let invT = 1.0 / Float(temperature)
            for i in 0..<vocabSize { logits[i] *= invT }
        }

        // top-k
        let k = max(1, cfg.topKCandidates)
        var idxs = Array(0..<vocabSize)
        idxs.sort { logits[$0] > logits[$1] }
        var kept = Array(idxs.prefix(min(Int(k), vocabSize)))

        // softmax over kept
        let maxLogit = kept.map { logits[$0] }.max() ?? 0
        var probs = kept.map { expf(logits[$0] - maxLogit) }
        var sumExp: Float = probs.reduce(0, +)
        if sumExp <= 0 { return CLlamaToken(kept.first ?? 0) }
        for i in 0..<probs.count { probs[i] /= sumExp }

        // top-p
        let topP = min(max(0.05, cfg.topProbabilityMass), 1.0)
        if topP < 0.999 {
            let order = (0..<kept.count).sorted { probs[$0] > probs[$1] }
            var cum: Float = 0
            var cut = kept.count
            for (rank, pos) in order.enumerated() {
                cum += probs[pos]
                if cum >= topP { cut = rank + 1; break }
            }
            let trimmed = Array(order.prefix(cut))
            var newKept: [Int] = []
            var newProbs: [Float] = []
            var newSum: Float = 0
            for pos in trimmed {
                newKept.append(kept[pos])
                let p = probs[pos]
                newProbs.append(p)
                newSum += p
            }
            if newSum > 0 {
                kept = newKept
                for i in 0..<newProbs.count { newProbs[i] /= newSum }
                probs = newProbs
            }
        }

        // multinomial draw
        let r = Float.random(in: 0..<1)
        var acc: Float = 0
        for (i, p) in probs.enumerated() {
            acc += p
            if r <= acc { return CLlamaToken(kept[i]) }
        }
        return CLlamaToken(kept.last ?? 0)
    }

    public func performShutdown() async {
        if let batch = clBatch { LlamaKitBridge.freeBatch(batch); clBatch = nil }
        if let context = clContext { LlamaKitBridge.freeContext(context); clContext = nil }
        if let model = clModel { LlamaKitBridge.freeModel(model); clModel = nil }

        currentContextTokens.removeAll()
        interruptFlag = false

        if loadingProgressContinuation != nil {
            publishProgress(LoadUpdate(stage: .failed, detail: "Shutdown initiated during loading.", hasError: true))
        }
        print("🦙 Kuzco LlamaInstance for \(profile.id) explicitly shut down 🦙")
    }

    private func performShutdownInternal() async {
        if let batch = clBatch { LlamaKitBridge.freeBatch(batch); clBatch = nil }
        if let context = clContext { LlamaKitBridge.freeContext(context); clContext = nil }
        if let model = clModel { LlamaKitBridge.freeModel(model); clModel = nil }
        currentContextTokens.removeAll()
        interruptFlag = false
        if loadingProgressContinuation != nil {
            publishProgress(LoadUpdate(stage: .failed, detail: "Shutdown due to error.", hasError: true))
        }
    }

    public func interruptCurrentPrediction() {
        interruptFlag = true
    }

    public func setGlobalPredictionConfig(_ newConfig: PredictionConfig) {
        self.predictionCfg = newConfig
    }

    public func setGlobalInstanceSettings(_ newSettings: InstanceSettings) {
        self.settings = newSettings
        print("Kuzco LlamaInstance (\(profile.id)) global instance settings updated. Context-related changes might require model reload.")
    }


public func generateTokenIDs(
    dialogue: [Turn],
    overrideSystemPrompt: String? = nil,
    overridePredictionConfig: PredictionConfig? = nil,
    overrideContextLength: UInt32? = nil
) -> AsyncThrowingStream<Int32, Error> {
    return AsyncThrowingStream { continuation in
        Task { @LlamaInstanceActor [] in
            do {
                for try await response in self.generateWithCompletionInfo(
                    dialogue: dialogue,
                    overrideSystemPrompt: overrideSystemPrompt,
                    overridePredictionConfig: overridePredictionConfig,
                    overrideContextLength: overrideContextLength
                ) {
                    // Poimi tokenId, vaikka content olisi tyhjä
                    if let tid = response.tokenId {
                        continuation.yield(tid)
                    }
                    if response.isComplete {
                        continuation.finish()
                        return
                    }
                }
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }
}



    public func generate(
        dialogue: [Turn],
        overrideSystemPrompt: String? = nil,
        overridePredictionConfig: PredictionConfig? = nil,
        overrideContextLength: UInt32? = nil
    ) -> AsyncThrowingStream<String, Error> {
        // Backward compatibility wrapper - just extracts content from StreamResponse
        return AsyncThrowingStream { continuation in
            Task { @LlamaInstanceActor [] in
                do {
                    for try await response in self.generateWithCompletionInfo(
                        dialogue: dialogue,
                        overrideSystemPrompt: overrideSystemPrompt,
                        overridePredictionConfig: overridePredictionConfig,
                        overrideContextLength: overrideContextLength
                    ) {
                        if !response.content.isEmpty {
                            continuation.yield(response.content)
                        }
                        if response.isComplete {
                            continuation.finish()
                            return
                        }
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func generateWithCompletionInfo(
        dialogue: [Turn],
        overrideSystemPrompt: String? = nil,
        overridePredictionConfig: PredictionConfig? = nil,
        overrideContextLength: UInt32? = nil
    ) -> AsyncThrowingStream<StreamResponse, Error> {
        interruptFlag = false


        let effectiveSystemPrompt = overrideSystemPrompt
        let effectivePredictionConfig = overridePredictionConfig ?? self.predictionCfg
        let currentCallContextLength = overrideContextLength ?? self.settings.contextLength

        let promptString: String
        if (overrideSystemPrompt == "speak"),
        let last = dialogue.last, last.role == .user {
            let voice = "tara"
            promptString = "<|audio|>\(voice): \(last.text)<|eot_id|>"
        } else {
            promptString = interactionFormatter.constructPrompt(
                for: dialogue,
                modelArchitecture: profile.architecture,
                systemPrompt: effectiveSystemPrompt
            )
        }
        // Dynamic budget for speak-mode: estimate number of 28-token windows from text length
        var speakText: String = ""
        var speakBudgetWindows: Int = Int.max
        if overrideSystemPrompt == "speak", let last = dialogue.last, last.role == .user {
            speakText = last.text
            // Heuristic tuned for SNAC @24kHz: ~0.085 s per window (28 ids)
            // Words-based: ~1.5 windows per word + small lead-in/out margin
            // Chars-based: ~1 window per ~9 chars + margin
            let words = speakText.split { $0.isWhitespace || $0.isNewline }.count
            let chars = speakText.count
            let byWords = Int(ceil(Double(words) * 2.2)) + 6
            let byChars = Int(ceil(Double(chars) / 9.0)) + 3
            // Use the larger estimate so we don't clip, but clamp to a sane upper bound
            speakBudgetWindows = max(4, min(24, max(byWords, byChars)))
            print("⏱️ Speak budget windows = \(speakBudgetWindows) (words=\(words), chars=\(chars), byWords=\(byWords), byChars=\(byChars))")
        }
        // ➋ SPEAK-tilassa: ei chat-templaten stoppeja, ei extra BOS/EOS
        var effectiveStopSequences: [String] = []
        var addBOSToken = true
        var addEOSToken = false
        if overrideSystemPrompt == "speak" {
            // Match python path: only stop at explicit EOT marker
            effectiveStopSequences = ["<|eot_id|>"]
            addBOSToken = false
            addEOSToken = false
            self.currentContextTokens.removeAll(keepingCapacity: false)
            self.lastAudioTokPerStep = Array(repeating: -1, count: 7)
        } else {
            // tavallisessa tilassa käytetään normaalit stop-sequt
            effectiveStopSequences = self.eosHandler.getAllEffectiveStopSequences()
        }

        return AsyncThrowingStream { continuation in
            Task { @LlamaInstanceActor [] in
                guard let model = self.clModel, let context = self.clContext, self.clBatch != nil else {
                    continuation.finish(throwing: KuzcoError.engineNotReady)
                    return
                }

                // Now that we have a valid `model`, compute vocab size and (for speak-mode) the boost helpers.
                let vocabSize = Int(LlamaKitBridge.getVocabSize(model: model))

                // Speak-mode helpers (soft bias towards audio codes and EOT)
                var speakBoostSet: Set<Int32> = []
                var speakEotTok: Int32? = nil
                if overrideSystemPrompt == "speak" {
                    let (boostSet, eotTok) = self.buildSpeakBoostSet(model: model)
                    speakBoostSet = boostSet
                    speakEotTok = eotTok
                }

                // Strict whitelist: vain validit <custom_token_*>, plus EOT-aliaset
                var speakWhitelist: Set<Int32> = []
                if overrideSystemPrompt == "speak" {
                    let (allowed, eotCanon) = self.buildCustomWhitelist(model: model)
                    speakWhitelist = allowed
                    if speakEotTok == nil { speakEotTok = eotCanon } // suositaan <|eot_id|>
                }
                // SPEAK-tilassa varmistetaan puhdas KV-cache
                if overrideSystemPrompt == "speak" {
                    LlamaKitBridge.clearKeyValueCache(context: context)
                    self.currentContextTokens.removeAll(keepingCapacity: false)
                    self.lastAudioTokPerStep = Array(repeating: -1, count: 7)
                }

                // Build (or reuse) the concrete ID cache for steps 0..6
                if overrideSystemPrompt == "speak" {
                    if self.speakStepCacheModelId != self.profile.id || self.speakStepIdCache[0].isEmpty {
                        self.speakStepIdCache = self.buildSpeakStepIdCache(model: model)
                        self.speakStepCacheModelId = self.profile.id
                        let counts = self.speakStepIdCache.map { $0.count }
                        print("🎛️ Speak step-id cache built: counts per step = \(counts)")
                    }
                }

                // Capture speak window budget for logging only (do NOT hard stop by it).
                let speakBudgetWindowsLocal = speakBudgetWindows

                // Safety cap independent of the text-derived estimate to avoid pathological loops.
                // Generous cap: allow up to ~64 windows worth of audio codes (~64 * 0.085 s ≈ 5.4 s).
                // We rely on <|eot_id|> to end naturally; this is only an emergency guard.
                let maxAudioTokensSpeak = 28 * 64
                let emergencyWindowCap = 64

                let actualMaxContextForThisCall = min(currentCallContextLength, self.settings.contextLength)

                do {
                    let promptTokens = try LlamaKitBridge.tokenize(text: promptString, model: model, addBos: addBOSToken, parseSpecial: true)

                    var commonPrefixLength = 0
                    while commonPrefixLength < self.currentContextTokens.count &&
                          commonPrefixLength < promptTokens.count &&
                          self.currentContextTokens[commonPrefixLength] == promptTokens[commonPrefixLength] {
                        commonPrefixLength += 1
                    }

                    if commonPrefixLength < self.currentContextTokens.count {
                        LlamaKitBridge.removeTokensFromKeyValueCache(context: context, sequenceId: 0, fromPosition: Int32(commonPrefixLength), toPosition: Int32(self.currentContextTokens.count))
                        print("🦙 Kuzco KV Cache: Shrunk from \(self.currentContextTokens.count) to \(commonPrefixLength) tokens 🦙")
                    }

                    let newTokensToProcess = Array(promptTokens.suffix(from: commonPrefixLength))
                    var kvCachePosition = Int32(commonPrefixLength)
                    self.currentContextTokens = promptTokens

                    if self.currentContextTokens.count >= actualMaxContextForThisCall {
                        let errMsg = "Prompt length (\(self.currentContextTokens.count)) exceeds effective context window for this call (\(actualMaxContextForThisCall)). Consider shortening history."
                        print("🦙 Kuzco Error: \(errMsg) 🦙")
                        LlamaKitBridge.clearKeyValueCache(context: context)
                        self.currentContextTokens.removeAll()
                        kvCachePosition = 0
                        continuation.finish(throwing: KuzcoError.configurationInvalid(reason: errMsg))
                        return
                    }

                    if !newTokensToProcess.isEmpty {
                        var evalIndex = 0
                        while evalIndex < newTokensToProcess.count {
                            guard self.clBatch != nil else { throw KuzcoError.batchCreationFailed }
                            LlamaKitBridge.clearBatch(&self.clBatch!)

                            var physicalBatchTokenCount: Int32 = 0

                            let batchEndIndex = min(evalIndex + Int(self.settings.processingBatchSize), newTokensToProcess.count)
                            for i in evalIndex..<batchEndIndex {
                                LlamaKitBridge.addTokenToBatch(batch: &self.clBatch!, token: newTokensToProcess[i], position: kvCachePosition + Int32(i - evalIndex), sequenceId: 0, enableLogits: false)
                                physicalBatchTokenCount += 1
                            }

                            guard self.clBatch != nil else { throw KuzcoError.batchCreationFailed }
                            self.clBatch!.logits[Int(physicalBatchTokenCount) - 1] = 1

                            LlamaKitBridge.setThreads(for: context, mainThreads: self.settings.cpuThreadCount, batchThreads: self.settings.cpuThreadCount)
                            try LlamaKitBridge.processBatch(context: context, batch: self.clBatch!)
                            kvCachePosition += physicalBatchTokenCount
                            evalIndex = batchEndIndex
                            if self.interruptFlag { 
                                continuation.yield(StreamResponse(content: "", isComplete: true, completionReason: .userStopped))
                                continuation.finish()
                                return
                            }
                        }
                    }

                    var generatedStringAccumulator = ""
                    var tokensGeneratedThisTurn = 0
                    // käytä aiemmin määriteltyjä stoppareita; SPEAK-tilassa tämä on tyhjä lista
                    let allStopSequences = effectiveStopSequences
                    let maxTokensForThisGeneration = effectivePredictionConfig.maxOutputTokens_effective(actualMaxContextForThisCall - UInt32(kvCachePosition))

                    var dbgCount = 0
                    // Counters for speak-mode gating
                    var audioCustomSeen = 0            // count of <custom_token_*> seen (for safety cap)
                    var validAudioAccepted = 0         // count of valid mapped SNAC ids (Python-style)
                    var unlocked = false               // becomes true right after first 28 valid codes

                    while tokensGeneratedThisTurn < maxTokensForThisGeneration {
                        var sampledToken: CLlamaToken = CLlamaToken(0)
                        if self.interruptFlag { 
                            continuation.yield(StreamResponse(content: "", isComplete: true, completionReason: .userStopped))
                            continuation.finish()
                            return
                        }
                        guard self.clBatch != nil else { throw KuzcoError.engineNotReady }
                        guard let logits = LlamaKitBridge.getLogitsOutput(context: context, fromBatchTokenIndex: self.clBatch!.n_tokens - 1 ) else {
                            throw KuzcoError.predictionFailed(details: "Failed to retrieve logits.")
                        }

                        if overrideSystemPrompt == "speak" {
                            // How many 7-id windows have we emitted *after* unlock?
                            let windowsSoFar = unlocked ? (1 + ((validAudioAccepted - 28) / 7)) : 0

                            // At a 7-token boundary?
                            let atBoundary =
                                unlocked &&
                                ((validAudioAccepted - 28) >= 0) &&
                                ((validAudioAccepted - 28) % 7) == 6

                            // Soft-stop policy:
                            //  • Start allowing EOT a little *before* the heuristic budget (soft start).
                            //  • Only *force* stop after a small grace beyond the budget.
                            let graceWindows = 8               // extra windows beyond heuristic budget
                            let eotSoftStartOffset = 3         // start allowing EOT 1 window before budget

                            let allowEOTNow = atBoundary && (windowsSoFar >= max(1, speakBudgetWindowsLocal - eotSoftStartOffset))
                            let forceStopNow = atBoundary && (windowsSoFar >= speakBudgetWindowsLocal + graceWindows)

                            if forceStopNow {
                                print("🛑 Speak: soft budget reached (\(windowsSoFar)/\(speakBudgetWindowsLocal)+\(graceWindows)) — stopping at boundary.")
                                continuation.yield(StreamResponse(content: "", isComplete: true, completionReason: .natural))
                                continuation.finish()
                                return
                            }

                            let step = validAudioAccepted % 7
                            let allowedNow = (step >= 0 && step < 7) ? self.speakStepIdCache[step] : []
                            sampledToken = self.maskedGreedyValidAudioForStep(
                                logitsPtr: logits,
                                vocabSize: vocabSize,
                                acceptedSoFar: validAudioAccepted,
                                allowedIds: allowedNow,
                                eotToken: speakEotTok,
                                allowEOT: allowEOTNow,
                                eotBoost: allowEOTNow ? 12.0 : 3.0
                            )
                        } else {
                            sampledToken = LlamaKitBridge.sampleTokenGreedy(model: model, context: context, logits: logits)
                        }



                    let piece = LlamaKitBridge.detokenize(token: sampledToken, model: model)

                    if overrideSystemPrompt == "speak" {
                        let isCustom = piece.hasPrefix("<custom_token_")
                        if isCustom {
                            audioCustomSeen += 1
                            if let mapped = self.snacId(fromCustomPiece: piece, acceptedSoFar: validAudioAccepted) {
                                validAudioAccepted += 1
                                if validAudioAccepted <= 35 {
                                    print("🔎 SNAC ok[\(validAudioAccepted-1)] = \(mapped)")
                                }
                                if !unlocked, validAudioAccepted == 28 {
                                    unlocked = true
                                    print("🔓 Speak: unlocked after first 28 valid SNAC codes (custom seen=\(audioCustomSeen))")
                                }
                                // Soft-stop policy:
                                //  - Encourage EOT near the heuristic budget (allowEOTNow with stronger eotBoost).
                                //  - Force-stop only after a small grace beyond the budget (see forceStopNow).
                                //  - Emergency caps remain as a last-resort guard.
                                if unlocked {
                                    let windowsSoFar = 1 + ((validAudioAccepted - 28) / 7)
                                    if windowsSoFar >= emergencyWindowCap { // ~64 * 0.085 s ≈ 5.4 s
                                        print("🛑 Speak: emergency cap reached (\(windowsSoFar) windows) — stopping.")
                                        continuation.yield(StreamResponse(content: "", isComplete: true, completionReason: .maxTokensReached))
                                        continuation.finish()
                                        return
                                    }
                                }
                            } else {
                                // Invalid custom piece for this step (e.g., small control code 1/2/3/4/5/6)
                                print("⚠️ snacId parse failed for piece '\(piece)' at accepted=\(validAudioAccepted)")
                            }

                            // Safety cap to avoid pathological loops even if valid codes don't accumulate
                            if audioCustomSeen >= maxAudioTokensSpeak {
                                print("🛑 Speak: audio token hard cap hit (\(audioCustomSeen) ≥ \(maxAudioTokensSpeak)) — stopping.")
                                continuation.yield(StreamResponse(content: "", isComplete: true, completionReason: .maxTokensReached))
                                continuation.finish()
                                return
                            }
                        }
                    }

                    if overrideSystemPrompt != "speak", LlamaKitBridge.isEndOfGenerationToken(model: model, token: sampledToken) {
                        continuation.yield(StreamResponse(content: "", isComplete: true, completionReason: .natural))
                        continuation.finish()
                        return
                    }

//                    let piece = LlamaKitBridge.detokenize(token: sampledToken, model: model)
                    var pieceToYield = piece
                    var stopForThisToken = false
                    var foundStopSequence: String? = nil
                    if overrideSystemPrompt == "speak" && dbgCount < 120 {
                        let isCustom = piece.hasPrefix("<custom_token_")
                        print("🎯 tokenId=\(sampledToken) piece='\(piece)' custom=\(isCustom)")
                        dbgCount += 1
                    }

                    // Soft nudge: if we're one token *before* a window boundary after unlock,
                    // bias toward emitting EOT by contributing the literal stop sequence via content check
                    if overrideSystemPrompt == "speak",
                       unlocked,
                       ((validAudioAccepted - 28) >= 0),
                       ((validAudioAccepted - 28) % 7) == 6,
                       let eot = speakEotTok {
                        // Slightly bias textual stop by appending nothing (sampling already has EOT boosted via speakBoostSet).
                        // No logits surgery here; the boosted sampler already prefers EOT as one of boosted tokens.
                        _ = eot // placeholder to silence 'unused' in case of different build flags
                    }

                    if !allStopSequences.isEmpty {
                        let checkBuffer = generatedStringAccumulator + piece
                        for stopSeq in allStopSequences {
                            if checkBuffer.hasSuffix(stopSeq) {
                                if let stopRangeInPiece = piece.range(of: stopSeq, options: [.anchored, .backwards], range: piece.startIndex..<piece.endIndex, locale: nil) {
                                    pieceToYield = String(piece[..<stopRangeInPiece.lowerBound])
                                } else if let stopRangeInAccumulator = (generatedStringAccumulator + piece).range(of: stopSeq, options: [.anchored, .backwards]) {
                                    if stopRangeInAccumulator.lowerBound < generatedStringAccumulator.endIndex {
                                        let pieceContributionStartIndex = max(generatedStringAccumulator.endIndex, stopRangeInAccumulator.lowerBound)
                                        let distanceIntoPiece = (generatedStringAccumulator + piece).distance(from: pieceContributionStartIndex, to: stopRangeInAccumulator.lowerBound)
                                        if distanceIntoPiece < 0 {
                                            let charsInPieceToCut = piece.distance(from: piece.startIndex, to: piece.index(piece.startIndex, offsetBy: -distanceIntoPiece))
                                            pieceToYield = charsInPieceToCut < piece.count ? String(piece.prefix(charsInPieceToCut)) : ""
                                        }
                                    }
                                }
                                stopForThisToken = true
                                foundStopSequence = stopSeq
                                print("🦙 Kuzco - Stopped by antiprompt '\(stopSeq)' (piece: \"\(piece)\", yielded: \"\(pieceToYield)\") 🦙")
                                break
                            }
                        }
                    }

                    // ⬇️ UUSI: aina emitataan tokenId (vaikka content olisi tyhjä)
                    // Huom. jos StreamResponse.tokenId on optional, poista Int32()-cast tai tee tokenId: Int32?
                    let resp = StreamResponse(
                        content: pieceToYield,
                        isComplete: false,
                        completionReason: nil,
                        tokenId: Int32(sampledToken)
                    )
                    continuation.yield(resp)

                    // Päivitä akkumulaattori alkuperäisellä piece:llä
                    generatedStringAccumulator += piece

                    if stopForThisToken {
                        // Lähetä päätösviesti (content tyhjä, tokenId ei tarpeen)
                        continuation.yield(StreamResponse(
                            content: "",
                            isComplete: true,
                            completionReason: .natural ,   // käytä teidän enumista sopivaa arvoa
                            tokenId: nil                   // jos tokenId on optional; jos ei, jätä pois parametri
                        ))
                        continuation.finish()
                        return
                    }

                        self.currentContextTokens.append(sampledToken)
                        guard self.clBatch != nil else { throw KuzcoError.batchCreationFailed }
                        LlamaKitBridge.clearBatch(&self.clBatch!)
                        LlamaKitBridge.addTokenToBatch(batch: &self.clBatch!, token: sampledToken, position: kvCachePosition, sequenceId: 0, enableLogits: true)

                        LlamaKitBridge.setThreads(for: context, mainThreads: self.settings.cpuThreadCount, batchThreads: self.settings.cpuThreadCount)
                        try LlamaKitBridge.processBatch(context: context, batch: self.clBatch!)

                        kvCachePosition += 1
                        tokensGeneratedThisTurn += 1

                        if kvCachePosition >= actualMaxContextForThisCall {
                            print("🦙 Kuzco - Context limit (\(actualMaxContextForThisCall)) reached during generation 🦙")
                            continuation.yield(StreamResponse(content: "", isComplete: true, completionReason: .contextWindowFull))
                            continuation.finish()
                            return
                        }
                        await Task.yield()
                    }
                    
                    // If we reach here, we hit the max tokens limit
                    continuation.yield(StreamResponse(content: "", isComplete: true, completionReason: .maxTokensReached))
                    continuation.finish()

                } catch let error as KuzcoError {
                    continuation.yield(StreamResponse(content: "", isComplete: true, completionReason: .error(error.localizedDescription)))
                    continuation.finish(throwing: error)
                } catch {
                    continuation.yield(StreamResponse(content: "", isComplete: true, completionReason: .error(error.localizedDescription)))
                    continuation.finish(throwing: KuzcoError.unknown(details: error.localizedDescription))
                }
            }
        }
    }



    // Speak-mode sampler: temperature + top-k/top-p + repetition penalty
private func sampleTokenSpeak(
    model: CLlamaModel,
    logitsPtr: UnsafePointer<Float>,
    vocabSize: Int,
    cfg: PredictionConfig,
    recentTokens: [CLlamaToken],
    repeatWindow: Int = 128
) -> CLlamaToken {
    // Kopioi logits Swift-taulukkoon
    var logits = Array(UnsafeBufferPointer(start: logitsPtr, count: vocabSize))

    // Repetition penalty
    if cfg.repetitionPenalty > 1.0, !recentTokens.isEmpty {
        let start = max(0, recentTokens.count - repeatWindow)
        for t in recentTokens[start...] {
            let idx = Int(t)
            if idx >= 0 && idx < vocabSize {
                let v = logits[idx]
                logits[idx] = v >= 0 ? (v / Float(cfg.repetitionPenalty)) : (v * Float(cfg.repetitionPenalty))
            }
        }
    }

    // Temperature
    let temperature = max(0.05, cfg.temperature)
    if temperature != 1.0 {
        let invT = 1.0 / Float(temperature)
        for i in 0..<vocabSize { logits[i] *= invT }
    }

    // top-k
    var idxs = Array(0..<vocabSize)
    let k = max(1, cfg.topKCandidates)
    idxs.sort { logits[$0] > logits[$1] }
    var kept = Array(idxs.prefix(min(Int(k), vocabSize)))

    // softmax kept-joukkoon
    let maxLogit = kept.map { logits[$0] }.max() ?? 0
    var probs: [Float] = []
    probs.reserveCapacity(kept.count)
    var sumExp: Float = 0
    for id in kept {
        let e = expf(logits[id] - maxLogit)
        probs.append(e)
        sumExp += e
    }
    if sumExp <= 0 { return CLlamaToken(0) }
    for i in 0..<probs.count { probs[i] /= sumExp }

    // top-p
    let topP = min(max(0.05, cfg.topProbabilityMass), 1.0)
    if topP < 0.999 {
        let order = (0..<kept.count).sorted { probs[$0] > probs[$1] }
        var cum: Float = 0
        var cut = kept.count
        for (rank, pos) in order.enumerated() {
            cum += probs[pos]
            if cum >= topP { cut = rank + 1; break }
        }
        let trimmed = Array(order.prefix(cut))
        var newKept: [Int] = []
        var newProbs: [Float] = []
        var newSum: Float = 0
        for pos in trimmed {
            newKept.append(kept[pos])
            let p = probs[pos]
            newProbs.append(p)
            newSum += p
        }
        if newSum > 0 {
            for i in 0..<newProbs.count { newProbs[i] /= newSum }
            kept = newKept
            probs = newProbs
        }
    }

    // multinomial sample
    let r = Float.random(in: 0..<1)
    var acc: Float = 0
    for (i, p) in probs.enumerated() {
        acc += p
        if r <= acc { return CLlamaToken(kept[i]) }
    }
    return CLlamaToken(kept.last ?? 0)
}
}

private extension PredictionConfig {
    func maxOutputTokens_effective(_ remainingContextLength: UInt32) -> Int {
        let limit = Int(remainingContextLength)
        guard limit > 0 else { return 0 }
        if maxOutputTokens == -1 { return limit }
        return min(maxOutputTokens, limit)
    }
}
