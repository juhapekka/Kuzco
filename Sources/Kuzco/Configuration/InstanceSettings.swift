//
//  InstanceSettings.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation

/// Settings for initializing and configuring a `LlamaInstance`.
public struct InstanceSettings: Codable, Hashable {
    public var contextLength: UInt32
    public var processingBatchSize: UInt32
    public var offloadedGpuLayers: Int32
    public var enableMemoryMapping: Bool
    public var enableMemoryLocking: Bool
    public var inferenceSeed: UInt32
    public var useFlashAttention: Bool
    public var cpuThreadCount: Int32

    public init(
        contextLength: UInt32 = 4096,
        processingBatchSize: UInt32 = 512,
        offloadedGpuLayers: Int32 = DeviceTuner.suggestedGpuLayers(),
        enableMemoryMapping: Bool = true,
        enableMemoryLocking: Bool = false,
        inferenceSeed: UInt32 = 0,
        useFlashAttention: Bool = false,
        cpuThreadCount: Int32 = DeviceTuner.suggestedCpuThreads()
    ) {
        self.contextLength = contextLength
        self.processingBatchSize = processingBatchSize
        self.offloadedGpuLayers = offloadedGpuLayers
        self.enableMemoryMapping = enableMemoryMapping
        self.enableMemoryLocking = enableMemoryLocking
        self.inferenceSeed = (inferenceSeed == 0) ? UInt32.random(in: 1...UInt32.max) : inferenceSeed
        self.useFlashAttention = useFlashAttention
        self.cpuThreadCount = cpuThreadCount
    }

    public static var standard: InstanceSettings {
        InstanceSettings()
    }

    public static var performanceFocused: InstanceSettings {
        #if os(iOS) || os(visionOS)
        return InstanceSettings(contextLength: 2048, processingBatchSize: 256, useFlashAttention: true)
        #else // macOS
        return InstanceSettings(contextLength: 4096, processingBatchSize: 512, useFlashAttention: true)
        #endif
    }
}
