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
    case unknown(details: String? = nil)

    public var errorDescription: String? {
        switch self {
        case .modelFileNotAccessible(let path):
            return "Model file is not accessible at: \(path)."
        case .modelInitializationFailed(let details):
            return "Failed to initialize model: \(details)."
        case .contextCreationFailed(let details):
            return "Failed to create inference context: \(details)."
        case .batchCreationFailed:
            return "Failed to create token batch."
        case .tokenizationFailed(let details):
            return "Tokenization process failed: \(details)."
        case .predictionFailed(let details):
            return "Text prediction failed: \(details)."
        case .engineNotReady:
            return "The LlamaInstance is not ready. Ensure a model is loaded."
        case .operationInterrupted:
            return "The current operation was interrupted."
        case .resourceDeallocated:
            return "Attempted to use a deallocated resource."
        case .configurationInvalid(let reason):
            return "The provided configuration is invalid: \(reason)."
        case .warmUpRoutineFailed(let details):
            return "The model warm-up routine failed: \(details)."
        case .unknown(let details):
            return "An unknown error occurred. \(details ?? "No additional details.")"
        }
    }
}
