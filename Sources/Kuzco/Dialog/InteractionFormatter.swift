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
        let mutableDialogue = dialogue

        // Always ensure we have a system prompt
        let effectiveSystemPrompt = systemPrompt ?? "You are a helpful AI assistant."
        
        // Add BOS token for LLaMA 3 if needed
        if modelArchitecture == .llama3 {
            prompt += llamaCpp.bosTokenString()
        }

        switch modelArchitecture {
        case .llamaGeneral, .codellama:
            prompt += "<<SYS>>\n\(effectiveSystemPrompt)\n<</SYS>>\n\n"
            for turn in mutableDialogue {
                if turn.role == .user {
                    prompt += "[INST] \(turn.text) [/INST]"
                } else if turn.role == .assistant {
                    prompt += " \(turn.text) "
                }
            }

        case .llama3:
            prompt += "<|start_header_id|>system<|end_header_id|>\n\n\(effectiveSystemPrompt)<|eot_id|>"
            for turn in mutableDialogue {
                prompt += "<|start_header_id|>\(turn.role.rawValue)<|end_header_id|>\n\n\(turn.text)<|eot_id|>"
            }
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        case .mistralInstruct, .mixtral:
            for turn in mutableDialogue {
                if turn.role == .user {
                    let userContent = turn == mutableDialogue.first ? "\(effectiveSystemPrompt)\n\(turn.text)" : turn.text
                    prompt += "[INST] \(userContent) [/INST]"
                } else if turn.role == .assistant {
                    prompt += "\(turn.text)</s>"
                }
            }

        case .phiGeneric:
            prompt += "<|system|>\n\(effectiveSystemPrompt)<|end|>\n"
            for turn in mutableDialogue {
                switch turn.role {
                case .user: prompt += "<|user|>\n\(turn.text)<|end|>\n"
                case .assistant: prompt += "<|assistant|>\n\(turn.text)<|end|>\n"
                case .system: break // Skip since we already added it
                }
            }
            prompt += "<|assistant|>\n"

        case .gemmaInstruct:
            for turn in mutableDialogue {
                let roleString = turn.role == .user ? "user" : "model"
                prompt += "<start_of_turn>\(roleString)\n"
                if turn.role == .user && turn == mutableDialogue.first {
                    prompt += "\(effectiveSystemPrompt)\n"
                }
                prompt += "\(turn.text)<end_of_turn>\n"
            }
            prompt += "<start_of_turn>model\n"

        case .openChat:
            prompt += "<|im_start|>system\n\(effectiveSystemPrompt)<|im_end|>\n"
            for turn in mutableDialogue {
                prompt += "<|im_start|>\(turn.role.rawValue)\n\(turn.text)<|im_end|>\n"
            }
            prompt += "<|im_start|>assistant\n"
            
        case .qwen2, .qwen3:
            prompt += "<|im_start|>system\n\(effectiveSystemPrompt)<|im_end|>\n"
            for turn in mutableDialogue {
                prompt += "<|im_start|>\(turn.role.rawValue)\n\(turn.text)<|im_end|>\n"
            }
            prompt += "<|im_start|>assistant\n"
            
        case .deepseek:
            prompt += "User: \(effectiveSystemPrompt)\n\n"
            for turn in mutableDialogue {
                if turn.role == .user {
                    prompt += "User: \(turn.text)\n\n"
                } else if turn.role == .assistant {
                    prompt += "Assistant: \(turn.text)\n\n"
                }
            }
            prompt += "Assistant: "
            
        case .commandR:
            prompt += "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\(effectiveSystemPrompt)<|END_OF_TURN_TOKEN|>"
            for turn in mutableDialogue {
                let roleToken = turn.role == .user ? "<|USER_TOKEN|>" : "<|CHATBOT_TOKEN|>"
                prompt += "<|START_OF_TURN_TOKEN|>\(roleToken)\(turn.text)<|END_OF_TURN_TOKEN|>"
            }
            prompt += "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            
        case .yi:
            prompt += "<|im_start|>system\n\(effectiveSystemPrompt)<|im_end|>\n"
            for turn in mutableDialogue {
                prompt += "<|im_start|>\(turn.role.rawValue)\n\(turn.text)<|im_end|>\n"
            }
            prompt += "<|im_start|>assistant\n"
            
        case .unknown:
            // Fallback to a generic ChatML format which is widely supported
            print("⚠️ Kuzco Warning: Unknown model architecture, using ChatML format as fallback")
            prompt += "<|im_start|>system\n\(effectiveSystemPrompt)<|im_end|>\n"
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
