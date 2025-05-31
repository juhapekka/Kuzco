//
//  LlamaKitBridge.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation
import llama

typealias CLlamaModel = OpaquePointer
typealias CLlamaContext = OpaquePointer
typealias CLlamaToken = llama_token
typealias CLlamaBatch = llama_batch

enum LlamaKitBridge {
    static func initializeLlamaBackend() {
        llama_backend_init()
    }

    static func releaseLlamaBackend() {
        llama_backend_free()
    }

    static func loadModel(from path: String, settings: InstanceSettings) throws -> CLlamaModel {
        var mparams = llama_model_default_params()
        mparams.n_gpu_layers = settings.offloadedGpuLayers
        mparams.use_mmap = settings.enableMemoryMapping
        mparams.use_mlock = settings.enableMemoryLocking

        guard let modelPtr = llama_load_model_from_file(path, mparams) else {
            throw KuzcoError.modelInitializationFailed(details: "llama_load_model_from_file returned null for path \(path).")
        }
        return modelPtr
    }

    static func freeModel(_ model: CLlamaModel) {
        llama_free_model(model)
    }

    static func createContext(for model: CLlamaModel, settings: InstanceSettings) throws -> CLlamaContext {
        var cparams = llama_context_default_params()
        cparams.n_ctx = settings.contextLength
        cparams.n_batch = settings.processingBatchSize
        cparams.n_ubatch = settings.processingBatchSize
        cparams.flash_attn = settings.useFlashAttention
        cparams.n_threads = settings.cpuThreadCount
        cparams.n_threads_batch = settings.cpuThreadCount

        guard let contextPtr = llama_new_context_with_model(model, cparams) else {
            throw KuzcoError.contextCreationFailed(details: "llama_new_context_with_model returned null.")
        }
        return contextPtr
    }

    static func freeContext(_ context: CLlamaContext) {
        llama_free(context)
    }

    static func getModelMaxContextLength(context: CLlamaContext) -> UInt32 {
        return llama_n_ctx(context)
    }

    static func getModelVocabularySize(model: CLlamaModel) -> Int32 {
        return llama_n_vocab(model)
    }

    static func tokenize(text: String, model: CLlamaModel, addBos: Bool, parseSpecial: Bool) throws -> [CLlamaToken] {
        let maxTokenCount = text.utf8.count + (addBos ? 1 : 0) + 1
        var tokens = [CLlamaToken](repeating: 0, count: maxTokenCount)

        let count = llama_tokenize(model, text, Int32(text.utf8.count), &tokens, Int32(maxTokenCount), addBos, parseSpecial)

        if count < 0 {
            throw KuzcoError.tokenizationFailed(details: "llama_tokenize returned \(count).")
        }
        return Array(tokens.prefix(Int(count)))
    }

    static func detokenize(token: CLlamaToken, model: CLlamaModel) -> String {
        let bufferSize = 128
        var buffer = [CChar](repeating: 0, count: bufferSize)

        let nChars = llama_token_to_piece(model, token, &buffer, Int32(bufferSize), 0, false)

        if nChars <= 0 {
            if nChars < 0 && -Int(nChars) > bufferSize {
                print("🦙 KuzcoBridge Error: Buffer too small for detokenizing token \(token). Required: \(-Int(nChars)), available: \(bufferSize) 🦙")
                return "<\(token_id_error_buffer_small)>"
            }

            return ""
        }

        let pieceBytes = buffer.prefix(Int(nChars)).map { UInt8(bitPattern: $0) }
        return String(decoding: pieceBytes, as: UTF8.self)
    }

    private static let token_id_error_buffer_small = -3

    private static let token_id_error = -1
    private static let token_unknown = -2


    // MARK: Batch Processing & KV Cache
    static func createBatch(maxTokens: UInt32, embeddingSize: Int32 = 0, numSequences: Int32 = 1) throws -> CLlamaBatch {
        let batch = llama_batch_init(Int32(maxTokens), embeddingSize, numSequences)
        return batch
    }

    static func freeBatch(_ batch: CLlamaBatch) {
        llama_batch_free(batch)
    }

    static func clearBatch(_ batch: inout CLlamaBatch) {
        batch.n_tokens = 0
    }

    static func addTokenToBatch(batch: inout CLlamaBatch, token: CLlamaToken, position: Int32, sequenceId: Int32, enableLogits: Bool) {
        let currentTokenIndex = batch.n_tokens

        batch.token[Int(currentTokenIndex)] = token
        batch.pos[Int(currentTokenIndex)] = llama_pos(position)
        batch.n_seq_id[Int(currentTokenIndex)] = 1

        batch.seq_id[Int(currentTokenIndex)]!.pointee = llama_seq_id(sequenceId)

        batch.logits[Int(currentTokenIndex)] = enableLogits ? 1 : 0
        batch.n_tokens += 1
    }

    static func setThreads(for context: CLlamaContext, mainThreads: Int32, batchThreads: Int32) {
        llama_set_n_threads(context, mainThreads, batchThreads)
    }

    static func processBatch(context: CLlamaContext, batch: CLlamaBatch) throws {
        let result = llama_decode(context, batch)
        if result != 0 {
            throw KuzcoError.predictionFailed(details: "llama_decode returned \(result).")
        }
    }

    static func getLogitsOutput(context: CLlamaContext, fromBatchTokenIndex index: Int32) -> UnsafeMutablePointer<Float>? {
        return llama_get_logits_ith(context, index)
    }
    
    static func clearKeyValueCache(context: CLlamaContext) {
        llama_kv_cache_clear(context)
    }

    static func removeTokensFromKeyValueCache(context: CLlamaContext, sequenceId: Int32, fromPosition start: Int32, toPosition end: Int32) {
        llama_kv_cache_seq_rm(context, llama_seq_id(sequenceId), llama_pos(start), llama_pos(end))
    }

    static func sampleTokenGreedy(model: CLlamaModel, context: CLlamaContext, logits: UnsafeMutablePointer<Float>) -> CLlamaToken {
        let vocabSize = llama_n_vocab(model)
        
        var maxLogit: Float = -Float.infinity
        var bestToken: CLlamaToken = 0
        
        for i in 0..<Int(vocabSize) {
            if logits[i] > maxLogit {
                maxLogit = logits[i]
                bestToken = CLlamaToken(i)
            }
        }
        return bestToken
    }

    static func getBosToken(model: CLlamaModel) -> CLlamaToken {
        return llama_token_bos(model)
    }

    static func getEosToken(model: CLlamaModel) -> CLlamaToken {
        return llama_token_eos(model)
    }

    static func isEndOfGenerationToken(model: CLlamaModel, token: CLlamaToken) -> Bool {
        return token == llama_token_eos(model) || llama_token_is_eog(model, token)
    }
}
