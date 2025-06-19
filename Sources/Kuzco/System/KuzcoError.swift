//
//  KuzcoError.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation

public enum KuzcoError: Error, LocalizedError, Equatable {
    case modelFileNotAccessible(path: String)
    case modelInitializationFailed(details: String)
    case contextCreationFailed(details: String)
    case batchCreationFailed
    case tokenizationFailed(details: String)
    case predictionFailed(details: String)
    case engineNotReady
    case operationInterrupted
    case resourceDeallocated
    case configurationInvalid(reason: String)
    case warmUpRoutineFailed(details: String)
    case unsupportedModelArchitecture(architecture: String, suggestedAction: String)
    case modelValidationFailed(reason: String)
    case insufficientMemory(details: String)
    case fallbackFailed(originalError: String, fallbackError: String)
    case unknown(details: String? = nil)

    public var errorDescription: String? {
        switch self {
        case .modelFileNotAccessible(let path):
            return "Model file is not accessible at: \(path). Please check the file path and permissions."
        case .modelInitializationFailed(let details):
            return "Failed to initialize model: \(details)"
        case .contextCreationFailed(let details):
            return "Failed to create inference context: \(details)"
        case .batchCreationFailed:
            return "Failed to create token batch. This may indicate insufficient memory or invalid batch parameters."
        case .tokenizationFailed(let details):
            return "Tokenization process failed: \(details)"
        case .predictionFailed(let details):
            return "Text prediction failed: \(details)"
        case .engineNotReady:
            return "The LlamaInstance is not ready. Ensure a model is loaded and initialized properly."
        case .operationInterrupted:
            return "The current operation was interrupted by user request."
        case .resourceDeallocated:
            return "Attempted to use a deallocated resource. This indicates a programming error."
        case .configurationInvalid(let reason):
            return "The provided configuration is invalid: \(reason)"
        case .warmUpRoutineFailed(let details):
            return "The model warm-up routine failed: \(details). The model may still work but performance could be affected."
        case .unsupportedModelArchitecture(let architecture, let suggestedAction):
            return "Unsupported model architecture '\(architecture)'. \(suggestedAction)"
        case .modelValidationFailed(let reason):
            return "Model file validation failed: \(reason). Please ensure you're using a valid GGUF model file."
        case .insufficientMemory(let details):
            return "Insufficient memory to complete operation: \(details). Try reducing context length or batch size."
        case .fallbackFailed(let originalError, let fallbackError):
            return "Both primary and fallback attempts failed. Original: \(originalError). Fallback: \(fallbackError)"
        case .unknown(let details):
            return "An unknown error occurred. \(details ?? "No additional details available.")"
        }
    }
    
    /// Returns true if this error might be recoverable with different settings
    public var isRecoverable: Bool {
        switch self {
        case .unsupportedModelArchitecture, .contextCreationFailed, .insufficientMemory:
            return true
        case .modelFileNotAccessible, .modelValidationFailed, .configurationInvalid:
            return false
        default:
            return false
        }
    }
    
    /// Provides suggestions for recovering from this error
    public var recoverySuggestion: String? {
        switch self {
        case .unsupportedModelArchitecture(_, let suggestion):
            return suggestion
        case .contextCreationFailed:
            return "Try reducing context length, batch size, or disabling GPU acceleration."
        case .insufficientMemory:
            return "Reduce context length, batch size, or close other memory-intensive applications."
        case .modelValidationFailed:
            return "Ensure you're using a valid GGUF model file from a trusted source."
        case .configurationInvalid:
            return "Check your model configuration and ensure all parameters are within valid ranges."
        default:
            return nil
        }
    }
}
