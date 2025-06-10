//
//  CompletionReason.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation

/// Indicates why text generation completed
public enum CompletionReason: Equatable, Codable {
    case natural              // Model hit EOS token and finished naturally
    case maxTokensReached     // Hit the max_tokens limit
    case contextWindowFull    // Ran out of context window space
    case userStopped         // User called stopCurrentGeneration()
    case error(String)       // Generation error with description
    case stopSequenceFound(String) // Hit a custom stop sequence
    
    /// Returns true if this completion reason indicates the user might want to continue generating
    public var shouldOfferContinuation: Bool {
        switch self {
        case .maxTokensReached, .contextWindowFull:
            return true
        case .natural, .userStopped, .error, .stopSequenceFound:
            return false
        }
    }
} 