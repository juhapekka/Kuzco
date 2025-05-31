//
//  InteractionFormatter.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation

public protocol InteractionFormatting {
    func constructPrompt(for dialogue: [Turn], modelArchitecture: ModelArchitecture, systemPrompt: String?) -> String
}

public struct StandardInteractionFormatter: InteractionFormatting {
    
    public init() {}

    public func constructPrompt(for dialogue: [Turn], modelArchitecture: ModelArchitecture, systemPrompt: String? = nil) -> String {
        var prompt = ""
        var mutableDialogue = dialogue

        if modelArchitecture != .gemmaInstruct && mutableDialogue.first?.role != .system {
            if modelArchitecture == .llama3 { prompt += llamaCpp.bosTokenString() }
            mutableDialogue.insert(Turn(role: .system, text: systemPrompt ?? "You are a helpful AI assistant."), at: 0)
        } else if modelArchitecture == .llama3 && dialogue.first?.role == .system {
             prompt += llamaCpp.bosTokenString()
        }


        switch modelArchitecture {
        case .llamaGeneral:
            var currentSystem: String?
            if mutableDialogue.first?.role == .system {
                currentSystem = "<<SYS>>\n\(mutableDialogue.removeFirst().text)\n<</SYS>>\n\n"
            }

            for turn in mutableDialogue {
                if turn.role == .user {
                    if let sys = currentSystem {
                        prompt += sys
                        currentSystem = nil
                    }
                    prompt += "[INST] \(turn.text) [/INST]"
                } else if turn.role == .assistant {
                    prompt += " \(turn.text) "
                }
            }
            if mutableDialogue.last?.role == .user {
            }


        case .llama3:
            for turn in mutableDialogue {
                prompt += "<|start_header_id|>\(turn.role.rawValue)<|end_header_id|>\n\n\(turn.text)<|eot_id|>"
            }
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        case .mistralInstruct:
            var systemInstruction: String?
            if mutableDialogue.first?.role == .system {
                systemInstruction = mutableDialogue.removeFirst().text
            }

            for turn in mutableDialogue {
                if turn.role == .user {
                    var userContent = ""
                    if let sys = systemInstruction {
                        userContent += "\(sys)\n"
                        systemInstruction = nil
                    }
                    userContent += turn.text
                    prompt += "[INST] \(userContent) [/INST]"
                } else if turn.role == .assistant {
                    prompt += "\(turn.text)</s>"
                }
            }

        case .phiGeneric:
            for turn in mutableDialogue {
                switch turn.role {
                case .system: prompt += "<|system|>\n\(turn.text)<|end|>\n"
                case .user: prompt += "<|user|>\n\(turn.text)<|end|>\n"
                case .assistant: prompt += "<|assistant|>\n\(turn.text)<|end|>\n"
                }
            }
            prompt += "<|assistant|>\n"

        case .gemmaInstruct:
            var systemContentForGemma: String?
            if mutableDialogue.first?.role == .system {
                systemContentForGemma = mutableDialogue.removeFirst().text
            }
            for turn in mutableDialogue {
                let roleString = turn.role == .user ? "user" : "model"
                prompt += "<start_of_turn>\(roleString)\n"
                if turn.role == .user, let sys = systemContentForGemma {
                    prompt += "\(sys)\n"
                    systemContentForGemma = nil
                }
                prompt += "\(turn.text)<end_of_turn>\n"
            }
            prompt += "<start_of_turn>model\n"

        case .openChat:
            for turn in mutableDialogue {
                prompt += "<|im_start|>\(turn.role.rawValue)\n\(turn.text)<|im_end|>\n"
            }
            prompt += "<|im_start|>assistant\n"
        }
        return prompt
    }

    private enum llamaCpp {
        static func bosTokenString() -> String { "<|begin_of_text|>" }
    }
}
