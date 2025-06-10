# Continue Generating Feature

The Kuzco package now supports intelligent "continue generating" functionality through enhanced streaming responses that include completion information.

## New Types

### `CompletionReason`
Indicates why text generation stopped:
- `.natural` - Model hit EOS token and finished naturally
- `.maxTokensReached` - Hit the max_tokens limit  
- `.contextWindowFull` - Ran out of context window space
- `.userStopped` - User called `interruptCurrentPrediction()`
- `.error(String)` - Generation error with description
- `.stopSequenceFound(String)` - Hit a custom stop sequence

### `StreamResponse`
Enhanced streaming response that includes:
- `content: String` - The token content
- `isComplete: Bool` - Whether this is the final token
- `completionReason: CompletionReason?` - Only set when `isComplete = true`

## Usage

### Using the Enhanced Stream (Recommended)

```swift
for try await response in kuzco.predictWithCompletionInfo(
    dialogue: conversation,
    with: modelProfile
) {
    if response.isComplete {
        switch response.completionReason {
        case .natural:
            // Generation completed naturally - hide continue button
            assistantMsg.completionStatus = .completed
        case .maxTokensReached, .contextWindowFull:
            // Show "continue generating" button
            assistantMsg.completionStatus = .cutOff
        case .userStopped:
            assistantMsg.completionStatus = .manuallyStopped
        case .error(let description):
            assistantMsg.completionStatus = .error
            print("Generation error: \(description)")
        case .stopSequenceFound(let sequence):
            // Custom stop sequence found - usually means completion
            assistantMsg.completionStatus = .completed
        }
    } else {
        // Regular token, append to message
        buffer += response.content
    }
}
```

### Quick Check for Continue Button

```swift
if let reason = response.completionReason, response.isComplete {
    showContinueButton = reason.shouldOfferContinuation
}
```

### Backward Compatibility

The original `predict()` method still works exactly as before:

```swift
for try await token in kuzco.predict(dialogue: conversation, with: modelProfile) {
    buffer += token
}
```

## Migration Guide

1. **Existing code continues to work** - No breaking changes
2. **Gradual migration** - Switch to `predictWithCompletionInfo()` when ready
3. **Enhanced UX** - Use completion reasons to show appropriate UI states

## Implementation Notes

- The enhanced stream method (`generateWithCompletionInfo`) is the primary implementation
- The original method is now a wrapper that extracts just the content
- All completion conditions are properly detected and reported
- Thread-safe and maintains all existing performance characteristics 