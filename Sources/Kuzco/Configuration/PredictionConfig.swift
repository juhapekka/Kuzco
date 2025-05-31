//
//  PredictionConfig.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation

public struct PredictionConfig: Codable, Hashable {
    public var temperature: Float           // Controls randomness
    public var topKCandidates: Int32        // Top-K sampling
    public var topProbabilityMass: Float    // Top-P (nucleus) sampling
    public var minProbability: Float        // Min-P sampling
    public var typicalProbability: Float    // Locally typical sampling (tfs_z)
    public var repetitionPenalty: Float
    public var repetitionContextSize: Int32 // How many recent tokens to consider for penalty
    public var frequencyPenalty: Float
    public var presencePenalty: Float
    public var mirostatMode: Int32          // 0=off, 1=Mirostat v1, 2=Mirostat v2
    public var mirostatLearningRate: Float  // eta
    public var mirostatTargetEntropy: Float // tau
    public var maxOutputTokens: Int         // Max tokens to generate (-1 for indefinite until EOS or context full)

    public init(
        temperature: Float = 0.7,
        topKCandidates: Int32 = 40,
        topProbabilityMass: Float = 0.9,
        minProbability: Float = 0.05,
        typicalProbability: Float = 1.0,
        repetitionPenalty: Float = 1.1,
        repetitionContextSize: Int32 = 64,
        frequencyPenalty: Float = 0.0,
        presencePenalty: Float = 0.0,
        mirostatMode: Int32 = 0,
        mirostatLearningRate: Float = 0.1,
        mirostatTargetEntropy: Float = 5.0,
        maxOutputTokens: Int = -1
    ) {
        self.temperature = temperature
        self.topKCandidates = topKCandidates
        self.topProbabilityMass = topProbabilityMass
        self.minProbability = minProbability
        self.typicalProbability = typicalProbability
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.mirostatMode = mirostatMode
        self.mirostatLearningRate = mirostatLearningRate
        self.mirostatTargetEntropy = mirostatTargetEntropy
        self.maxOutputTokens = maxOutputTokens
    }

    public static var balanced: PredictionConfig {
        PredictionConfig()
    }

    public static var creative: PredictionConfig {
        PredictionConfig(temperature: 0.9, topKCandidates: 0, topProbabilityMass: 0.95, repetitionPenalty: 1.05)
    }

    public static var precise: PredictionConfig {
        PredictionConfig(temperature: 0.2, topKCandidates: 10, topProbabilityMass: 0.7, repetitionPenalty: 1.2)
    }
}
