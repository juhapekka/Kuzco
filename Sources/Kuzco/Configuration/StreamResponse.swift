//
//  StreamResponse.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation

/// Response structure for streaming generation that includes completion information
public struct StreamResponse {
    public let content: String           // The token content
    public let isComplete: Bool          // Whether this is the final token
    public let completionReason: CompletionReason?  // Only set when isComplete = true
    
    public init(content: String, isComplete: Bool = false, completionReason: CompletionReason? = nil) {
        self.content = content
        self.isComplete = isComplete
        self.completionReason = completionReason
    }
} 