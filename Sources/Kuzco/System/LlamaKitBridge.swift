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

    public static func getVocabSize(model: CLlamaModel) -> Int {
        return Int(llama_n_vocab(model))
    }

    /// Validates a model file before attempting to load it
    static func validateModelFile(path: String) throws {
        // Check if file exists
        guard FileManager.default.fileExists(atPath: path) else {
            throw KuzcoError.modelFileNotAccessible(path: path)
        }
        
        // Check file size (must be at least 1MB for a valid model)
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: path)
            if let fileSize = attributes[.size] as? UInt64, fileSize < 1_048_576 {
                throw KuzcoError.modelInitializationFailed(details: "Model file is too small (\(fileSize) bytes). Minimum expected size is 1MB.")
            }
        } catch {
            throw KuzcoError.modelFileNotAccessible(path: path)
        }
        
        // Validate GGUF magic bytes
        guard let fileHandle = FileHandle(forReadingAtPath: path) else {
            throw KuzcoError.modelFileNotAccessible(path: path)
        }
        
        defer { fileHandle.closeFile() }
        
        do {
            let magicBytes = fileHandle.readData(ofLength: 4)
            let expectedMagic = "GGUF".data(using: .ascii)!
            
            if magicBytes.count < 4 || magicBytes != expectedMagic {
                throw KuzcoError.configurationInvalid(reason: "Model file does not appear to be a valid GGUF format. Expected magic bytes 'GGUF', but found different format.")
            }
        } catch {
            throw KuzcoError.modelInitializationFailed(details: "Failed to read model file header: \(error.localizedDescription)")
        }
    }

    static func loadModel(from path: String, settings: InstanceSettings) throws -> CLlamaModel {
        // Pre-validate the model file
        try validateModelFile(path: path)
        
        // Wrap the actual loading in error handling
        do {
            var mparams = llama_model_default_params()
            mparams.n_gpu_layers = settings.offloadedGpuLayers
            mparams.use_mmap = settings.enableMemoryMapping
            mparams.use_mlock = settings.enableMemoryLocking

            print("🦙 Kuzco - Attempting to load model from: \(path) 🦙")
            print("🦙 GPU layers: \(settings.offloadedGpuLayers), mmap: \(settings.enableMemoryMapping), mlock: \(settings.enableMemoryLocking) 🦙")
            
            guard let modelPtr = llama_load_model_from_file(path, mparams) else {
                // Enhanced error handling with architecture detection
                let fileName = (path as NSString).lastPathComponent.lowercased()
                
                if fileName.contains("qwen3") || fileName.contains("qwen2") {
                    let architecture = fileName.contains("qwen3") ? "qwen3" : "qwen2"
                    throw KuzcoError.unsupportedModelArchitecture(
                        architecture: architecture,
                        suggestedAction: "Your version of llama.cpp doesn't support \(architecture) architecture. The model will still work but may use fallback formatting. Consider updating llama.cpp or using a compatible model format. Kuzco will attempt to handle this model with ChatML formatting."
                    )
                } else if fileName.contains("deepseek") {
                    throw KuzcoError.unsupportedModelArchitecture(
                        architecture: "deepseek",
                        suggestedAction: "DeepSeek models require specific llama.cpp support. Try using a different model or updating to a more recent version of llama.cpp."
                    )
                } else if fileName.contains("claude") || fileName.contains("gpt") {
                    throw KuzcoError.configurationInvalid(reason: "This appears to be a commercial model format that cannot be loaded with llama.cpp. Please use a GGUF-format open source model.")
                } else {
                    throw KuzcoError.modelInitializationFailed(details: "llama_load_model_from_file returned null for path \(path). This could be due to: 1) Unsupported model architecture, 2) Corrupted model file, 3) Insufficient memory, 4) Invalid GGUF format. Try using a different model or check the file integrity.")
                }
            }
            
            print("🦙 Kuzco - Model loaded successfully 🦙")
            return modelPtr
            
        } catch let error as KuzcoError {
            // Re-throw our custom errors
            throw error
        } catch {
            // Catch any unexpected errors from llama.cpp
            throw KuzcoError.modelInitializationFailed(details: "Unexpected error during model loading: \(error.localizedDescription)")
        }
    }

    /// Attempts to load a model with fallback approaches for unsupported architectures
    static func loadModelWithFallback(from path: String, settings: InstanceSettings, fallbackArchitecture: ModelArchitecture? = nil) throws -> CLlamaModel {
        do {
            return try loadModel(from: path, settings: settings)
        } catch let error as KuzcoError {
            print("🦙 Kuzco - Primary model load failed: \(error.localizedDescription) 🦙")
            
            // Try with reduced GPU layers as fallback
            if case .unsupportedModelArchitecture = error, settings.offloadedGpuLayers > 0 {
                print("🦙 Kuzco - Attempting fallback with CPU-only processing 🦙")
                var fallbackSettings = settings
                fallbackSettings.offloadedGpuLayers = 0
                
                do {
                    return try loadModel(from: path, settings: fallbackSettings)
                } catch {
                    print("🦙 Kuzco - Fallback also failed 🦙")
                    throw error
                }
            }
            
            // For unsupported architecture errors, we'll let the higher level handle fallback
            throw error
        }
    }

    static func freeModel(_ model: CLlamaModel) {
        do {
            llama_free_model(model)
            print("🦙 Kuzco - Model freed successfully 🦙")
        } catch {
            print("🦙 Kuzco - Warning: Error freeing model: \(error.localizedDescription) 🦙")
        }
    }

    static func createContext(for model: CLlamaModel, settings: InstanceSettings) throws -> CLlamaContext {
        do {
            var cparams = llama_context_default_params()
            cparams.n_ctx = settings.contextLength
            cparams.n_batch = settings.processingBatchSize
            cparams.n_ubatch = settings.processingBatchSize
            cparams.flash_attn = settings.useFlashAttention
            cparams.n_threads = settings.cpuThreadCount
            cparams.n_threads_batch = settings.cpuThreadCount

            print("🦙 Kuzco - Creating context with: ctx=\(settings.contextLength), batch=\(settings.processingBatchSize), threads=\(settings.cpuThreadCount) 🦙")

            guard let contextPtr = llama_new_context_with_model(model, cparams) else {
                throw KuzcoError.contextCreationFailed(details: "llama_new_context_with_model returned null. This may be due to insufficient memory or invalid context parameters.")
            }
            
            print("🦙 Kuzco - Context created successfully 🦙")
            return contextPtr
            
        } catch let error as KuzcoError {
            throw error
        } catch {
            throw KuzcoError.contextCreationFailed(details: "Unexpected error during context creation: \(error.localizedDescription)")
        }
    }

    static func freeContext(_ context: CLlamaContext) {
        do {
            llama_free(context)
            print("🦙 Kuzco - Context freed successfully 🦙")
        } catch {
            print("🦙 Kuzco - Warning: Error freeing context: \(error.localizedDescription) 🦙")
        }
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
            // Add more detailed error information
            let errorDetails = "llama_tokenize returned \(count) for text length \(text.utf8.count), maxTokens: \(maxTokenCount), addBos: \(addBos), parseSpecial: \(parseSpecial)"
            print("🦙 KuzcoBridge Tokenization Error: \(errorDetails) 🦙")
            throw KuzcoError.tokenizationFailed(details: errorDetails)
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
        do {
            let result = llama_decode(context, batch)
            if result != 0 {
                let errorMsg = "llama_decode returned \(result). This may indicate insufficient memory, invalid batch, or model corruption."
                print("🦙 KuzcoBridge Batch Processing Error: \(errorMsg) 🦙")
                throw KuzcoError.predictionFailed(details: errorMsg)
            }
        } catch let error as KuzcoError {
            throw error
        } catch {
            throw KuzcoError.predictionFailed(details: "Unexpected error during batch processing: \(error.localizedDescription)")
        }
    }

    static func getLogitsOutput(context: CLlamaContext, fromBatchTokenIndex index: Int32) -> UnsafeMutablePointer<Float>? {
        do {
            return llama_get_logits_ith(context, index)
        } catch {
            print("🦙 KuzcoBridge Warning: Error getting logits: \(error.localizedDescription) 🦙")
            return nil
        }
    }
    
    static func clearKeyValueCache(context: CLlamaContext) {
        do {
            llama_kv_cache_clear(context)
            print("🦙 Kuzco - KV cache cleared successfully 🦙")
        } catch {
            print("🦙 Kuzco - Warning: Error clearing KV cache: \(error.localizedDescription) 🦙")
        }
    }

    static func removeTokensFromKeyValueCache(context: CLlamaContext, sequenceId: Int32, fromPosition start: Int32, toPosition end: Int32) {
        do {
            llama_kv_cache_seq_rm(context, llama_seq_id(sequenceId), llama_pos(start), llama_pos(end))
        } catch {
            print("🦙 Kuzco - Warning: Error removing tokens from KV cache: \(error.localizedDescription) 🦙")
        }
    }

    static func sampleTokenGreedy(model: CLlamaModel, context: CLlamaContext, logits: UnsafeMutablePointer<Float>) -> CLlamaToken {
        do {
            let vocabSize = llama_n_vocab(model)
            
            guard vocabSize > 0 else {
                print("🦙 KuzcoBridge Error: Invalid vocabulary size: \(vocabSize) 🦙")
                return 0
            }
            
            var maxLogit: Float = -Float.infinity
            var bestToken: CLlamaToken = 0
            
            for i in 0..<Int(vocabSize) {
                if logits[i] > maxLogit {
                    maxLogit = logits[i]
                    bestToken = CLlamaToken(i)
                }
            }
            return bestToken
        } catch {
            print("🦙 KuzcoBridge Error: Exception during token sampling: \(error.localizedDescription) 🦙")
            return 0
        }
    }

    static func getBosToken(model: CLlamaModel) -> CLlamaToken {
        do {
            return llama_token_bos(model)
        } catch {
            print("🦙 KuzcoBridge Warning: Error getting BOS token: \(error.localizedDescription) 🦙")
            return 1 // Common fallback BOS token ID
        }
    }

    static func getEosToken(model: CLlamaModel) -> CLlamaToken {
        do {
            return llama_token_eos(model)
        } catch {
            print("🦙 KuzcoBridge Warning: Error getting EOS token: \(error.localizedDescription) 🦙")
            return 2 // Common fallback EOS token ID
        }
    }

    static func isEndOfGenerationToken(model: CLlamaModel, token: CLlamaToken) -> Bool {
        do {
            return token == llama_token_eos(model) || llama_token_is_eog(model, token)
        } catch {
            print("🦙 KuzcoBridge Warning: Error checking EOG token: \(error.localizedDescription) 🦙")
            return token == 2 // Fallback check for common EOS token
        }
    }
}
