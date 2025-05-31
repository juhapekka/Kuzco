//
//  DeviceTuner.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation

public enum DeviceTuner {
    public static func suggestedGpuLayers() -> Int32 {
        #if targetEnvironment(simulator)
            return 0
        #else
            return 99
        #endif
    }

    public static func suggestedCpuThreads() -> Int32 {
        let coreCount = ProcessInfo.processInfo.processorCount
        return Int32(max(1, coreCount > 2 ? coreCount - 2 : 1))
    }
}
