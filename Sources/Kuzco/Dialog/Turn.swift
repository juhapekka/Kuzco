//
//  Turn.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation

public enum DialogueRole: String, Codable, Hashable {
    case system
    case user
    case assistant
}

public struct Turn: Identifiable, Codable, Hashable {
    public let id: UUID
    public let role: DialogueRole
    public var text: String

    public init(id: UUID = UUID(), role: DialogueRole, text: String) {
        self.id = id
        self.role = role
        self.text = text
    }
}
