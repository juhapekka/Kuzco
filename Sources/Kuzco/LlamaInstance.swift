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

    // Build whitelist for speak-mode: allow only <custom_token_N> and (optionally) <|eot_id|>
    private func buildCustomWhitelist(model: CLlamaModel) -> (allowed: Set<Int32>, eot: Int32?) {
        let nVocab = Int(LlamaKitBridge.getVocabSize(model: model))
        var allowed = Set<Int32>()
        allowed.reserveCapacity(8192)
        var eotTok: Int32? = nil

        for tid in 0..<nVocab {
            let s = LlamaKitBridge.detokenize(token: Int32(tid), model: model)
            if s == "<|eot_id|>" {
                eotTok = Int32(tid)
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
        if (overrideSystemPrompt == "speak"),   // käytä tätä lippuna audio-tilaan
        let last = dialogue.last, last.role == .user {
            let voice = "tara" // vaihda haluamasi ääni tähän tai välitä parametrina
            promptString = "<|audio|>\(voice): \(last.text)<|eot_id|>"

            
        } else {
            promptString = interactionFormatter.constructPrompt(
                for: dialogue,
                modelArchitecture: profile.architecture,
                systemPrompt: effectiveSystemPrompt
            )
        }
        // ➋ SPEAK-tilassa: ei chat-templaten stoppeja, ei extra BOS/EOS
        var effectiveStopSequences: [String] = []
        var addBOSToken = true
        var addEOSToken = false
        if overrideSystemPrompt == "speak" {
            effectiveStopSequences = []     // ei antiprompteja
            addBOSToken = false             // älä lisää BOS
            addEOSToken = false             // älä pakota EOS:ia
            // poistetaan myös paikallinen kontekstilista (KV nollataan myöhemmin kun context on käytettävissä)
            self.currentContextTokens.removeAll(keepingCapacity: false)
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
                // SPEAK-tilassa varmistetaan puhdas KV-cache
                if overrideSystemPrompt == "speak" {
                    LlamaKitBridge.clearKeyValueCache(context: context)
                    self.currentContextTokens.removeAll(keepingCapacity: false)
                }

                // Build whitelist once for this speak-session
                var speakWhitelist: Set<Int32> = []
                var speakEOT: Int32? = nil
                if overrideSystemPrompt == "speak" {
                    let wl = self.buildCustomWhitelist(model: model)
                    speakWhitelist = wl.allowed
                    speakEOT = wl.eot
                    print("🔒 Speak whitelist size = \(speakWhitelist.count)")
                }

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

                    while tokensGeneratedThisTurn < maxTokensForThisGeneration {
                        if self.interruptFlag { 
                            continuation.yield(StreamResponse(content: "", isComplete: true, completionReason: .userStopped))
                            continuation.finish()
                            return
                        }
                        guard self.clBatch != nil else { throw KuzcoError.engineNotReady }
                        guard let logits = LlamaKitBridge.getLogitsOutput(context: context, fromBatchTokenIndex: self.clBatch!.n_tokens - 1 ) else {
                            throw KuzcoError.predictionFailed(details: "Failed to retrieve logits.")
                        }

                        let sampledToken: CLlamaToken
                        if overrideSystemPrompt == "speak" {
                            // Masked greedy over whitelist
                            let nVocab = Int(LlamaKitBridge.getVocabSize(model: model))
                            let buf = UnsafeBufferPointer(start: logits, count: nVocab)
                            var bestTok: Int32 = 0
                            var bestVal: Float = -Float.infinity
                            for i in 0..<nVocab {
                                let tid = Int32(i)
                                if speakWhitelist.contains(tid) {
                                    let v = buf[i]
                                    if v > bestVal { bestVal = v; bestTok = tid }
                                }
                            }
                            sampledToken = bestTok
                        } else {
                            sampledToken = LlamaKitBridge.sampleTokenGreedy(model: model, context: context, logits: logits)
                        }

                        // Explicit EOT guard for speak-mode
                        if overrideSystemPrompt == "speak", let eot = speakEOT, sampledToken == eot {
                            continuation.yield(StreamResponse(content: "", isComplete: true, completionReason: .natural, tokenId: Int32(sampledToken)))
                            continuation.finish()
                            return
                        }

                        if LlamaKitBridge.isEndOfGenerationToken(model: model, token: sampledToken) {
                            continuation.yield(StreamResponse(content: "", isComplete: true, completionReason: .natural))
                            continuation.finish()
                            return
                        }

                    let piece = LlamaKitBridge.detokenize(token: sampledToken, model: model)
                    var pieceToYield = piece
                    var stopForThisToken = false
                    var foundStopSequence: String? = nil
                    if overrideSystemPrompt == "speak" && dbgCount < 120 {
                        let isCustom = piece.hasPrefix("<custom_token_")
                        print("🎯 tokenId=\(sampledToken) piece='\(piece)' custom=\(isCustom)")
                        dbgCount += 1
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
