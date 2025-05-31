//
//  LoadUpdate.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation

public struct LoadUpdate: Equatable {
    public enum Stage: String, Equatable {
        case idle = "Idle"
        case preparing = "Preparing"
        case readingModel = "Reading Model Data"
        case creatingContext = "Initializing Context"
        case prewarming = "Pre-warming Engine"
        case ready = "Ready"
        case failed = "Failed"
    }

    public let stage: Stage
    public let detail: String?
    public let hasError: Bool

    public init(stage: Stage, detail: String? = nil, hasError: Bool = false) {
        self.stage = stage
        self.detail = detail
        self.hasError = hasError
    }

    public static var initial: LoadUpdate {
        .init(stage: .idle)
    }
}
