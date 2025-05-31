//
//  ModelProfile.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation

/// Specifies the type of LLM architecture for prompt formatting and behavior.
public enum ModelArchitecture: String, CaseIterable, Codable, Hashable {
    case llamaGeneral
    case llama3
    case mistralInstruct
    case phiGeneric
    case gemmaInstruct
    case openChat
}

public struct ModelProfile: Hashable, Codable {
    public let id: String
    public let sourcePath: String
    public let architecture: ModelArchitecture

    public init(id: String? = nil, sourcePath: String, architecture: ModelArchitecture) {
        self.id = id ?? sourcePath
        self.sourcePath = sourcePath
        self.architecture = architecture
    }
}
