@MainActor
private func continueGeneration(for assistantMsg: ChatMessage) async {
    guard llm.currentLoadPhase == .ready else { hud.showAlert("Model needs to load", image: .warning); return }
    guard !isLLMGenerating else { hud.showAlert("Please wait for the current response to finish.", image: .warning); return }
    
    // Find the message before this assistant message to get the user prompt
    guard let idx = session.messages.firstIndex(where: { $0.id == assistantMsg.id }),
          idx > 0, session.messages[idx - 1].role == .user else {
        hud.showAlert("Couldn't find original prompt.", image: .warning); return
    }
    let userPrompt = session.messages[idx - 1]
    
    // Build conversation history including the partial assistant response
    let historyUpToUser = Array(session.messages.prefix(upTo: idx))
    
    // Create a modified history that includes the partial response and asks for continuation
    var continuationHistory = historyUpToUser
    continuationHistory.append(ChatMessage(content: userPrompt.content, role: .user, session: session))
    continuationHistory.append(ChatMessage(content: assistantMsg.content, role: .assistant, session: session))
    
    // Add a simple continuation request
    let continuePrompt = "Continue"
    
    isTyping = true
    defer {
        isTyping = false
        #if os(iOS)
        triggerHaptic()
        #endif
    }

    var buffer = assistantMsg.content // Start with existing content
    var didTriggerStartHaptic = false

    do {
        for try await (content, isComplete, completionReason) in llm.stream(
            chatHistoryMessages: continuationHistory,
            userInput: continuePrompt,
            lengthPreference: nil
        ) {
            if !didTriggerStartHaptic && !content.isEmpty {
                #if os(iOS)
                triggerHaptic()
                #endif
                didTriggerStartHaptic = true
            }

            if !content.isEmpty {
                buffer += content
                assistantMsg.content = buffer
                assistantMsg.completionStatus = .unknown // Reset status during generation
                try? context.save()
            }
            
            // Handle completion
            if isComplete {
                if let reason = completionReason {
                    switch reason {
                    case .natural:
                        assistantMsg.completionStatus = .completed
                    case .maxTokensReached, .contextWindowFull:
                        assistantMsg.completionStatus = .cutOff
                    case .userStopped:
                        assistantMsg.completionStatus = .manuallyStopped
                    case .error:
                        assistantMsg.completionStatus = .cutOff
                    case .stopSequenceFound(_):
                        assistantMsg.completionStatus = .completed
                    }
                } else {
                    assistantMsg.completionStatus = .completed
                }
                try? context.save()
                break
            }
        }
    } catch {
        // Mark as cut off if there was an error
        assistantMsg.completionStatus = .cutOff
        try? context.save()
        hud.showAlert(error.localizedDescription, image: .failure)
    }
} 