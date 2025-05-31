# Kuzco 🦙

A Swift package for iOS (and macOS) that simplifies running local LLMs (Large Language Models) using `llama.cpp`. Designed for developers who want to integrate AI-assisted text generation directly into their Swift apps. 🧠✨

> **License**: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
> **Platforms**: iOS 15+, macOS 12+, Mac Catalyst 15+
> **Language**: Swift 5.9

---

## Features 🚀

- **Local LLM Execution**: Integrates with `llama.cpp` to run large language models directly on-device.
- **Flexible Configuration**: Tune model settings with `InstanceSettings` and `PredictionConfig`.
- **Customizable Prompt Formatting**: Supports multiple architectures (LLaMA, Mistral, Phi, Gemma, OpenChat) with customizable prompt formatting via `InteractionFormatting`.
- **Resource Management**: Efficient instance caching and automatic context handling.
- **Pre-Warming Engine**: Optional prewarming to optimize inference speed.
- **Async/Await Support**: Modern Swift concurrency for non-blocking prediction streams.
- **Cross-Platform**: Compatible with iOS, macOS, and Mac Catalyst.

---

## Getting Started 🏁

### Installation 📦

1.  In Xcode, go to **File > Add Packages**.
2.  Enter the repository URL: `https://github.com/your-username/Kuzco.git`
3.  Select the latest version and add the `Kuzco` package to your target.

### Usage 📝

```swift
import Kuzco

// Create a model profile
let profile = ModelProfile(
    sourcePath: "/path/to/your/model.gguf",
    architecture: .llama3
)

// Get Kuzco instance
let kuzco = Kuzco.shared

// Start a prediction
Task {
    do {
        let dialogue: [Turn] = [
            Turn(role: .user, text: "Hello!"),
        ]
        let stream = try await kuzco.predict(
            dialogue: dialogue,
            with: profile,
            instanceSettings: .performanceFocused,
            predictionConfig: .creative
        )
        for try await output in stream {
            print(output)
        }
    } catch {
        print("Error during prediction: \(error)")
    }
}
```

## Model Support 🧠

Supports multiple architectures with customized prompt formatting:

- **LLaMA General / LLaMA 3**
- **Mistral Instruct**
- **Phi Generic**
- **Gemma Instruct**
- **OpenChat**

## Tuning Parameters 🎛️

- **InstanceSettings**: Control context length, batch size, GPU offload layers, CPU threads, flash attention, and more.
- **PredictionConfig**: Fine-tune sampling strategies (temperature, top-K, top-P, repetition penalties, Mirostat, etc.).

## Contributing 🤝

Contributions are welcome! Please open an issue or submit a pull request.

## License 📜

Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
See `LICENSE` for details.

## Credits ✨

- Built on top of [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp), with Swift integration by [Jared Cassoutt](https://github.com/jaredcassoutt).
- Special thanks to the open-source community for making efficient LLMs on device a reality.
