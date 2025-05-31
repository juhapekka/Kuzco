//
//  EosHandler.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation

// Manages End-of-Speech (EOS) tokens and antiprompts for different model architectures.
public struct EosHandler {
    private let architecture: ModelArchitecture
    private var additionalStopSequences: [String]

    public init(architecture: ModelArchitecture, additionalStopSequences: [String] = []) {
        self.architecture = architecture
        self.additionalStopSequences = additionalStopSequences
    }

    public func getDefaultStopSequences() -> [String] {
        var sequences: [String] = []
        switch architecture {
        case .llamaGeneral: sequences.append(contentsOf: ["</s>", "[INST]"])
        case .llama3: sequences.append(contentsOf: ["<|eot_id|>", "<|start_header_id|>user<|end_header_id|>"])
        case .mistralInstruct: sequences.append(contentsOf: ["</s>", "[INST]"])
        case .phiGeneric: sequences.append(contentsOf: ["<|end|>", "<|user|>"])
        case .gemmaInstruct: sequences.append(contentsOf: ["<end_of_turn>", "<eos>", "<start_of_turn>user"])
        case .openChat: sequences.append(contentsOf: ["<|im_end|>", "<|im_start|>user"])
        }
        return sequences
    }

    public func getAllEffectiveStopSequences() -> [String] {
        return Array(Set(getDefaultStopSequences() + additionalStopSequences))
    }

    public mutating func addStopSequence(_ sequence: String) {
        if !additionalStopSequences.contains(sequence) {
            additionalStopSequences.append(sequence)
        }
    }
}
