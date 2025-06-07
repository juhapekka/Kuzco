# Kuzco 🦙

[![Swift Package Manager](https://img.shields.io/badge/Swift%20Package%20Manager-compatible-brightgreen.svg)](https://swift.org/package-manager/)
[![Platform](https://img.shields.io/badge/platform-iOS%2015%2B%20|%20macOS%2012%2B%20|%20Mac%20Catalyst%2015%2B-blue.svg)](https://swift.org)
[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**Kuzco** is a powerful, easy-to-use Swift package that brings local Large Language Model (LLM) inference to iOS and macOS apps. Built on top of the battle-tested `llama.cpp`, Kuzco enables you to run AI models directly on-device with zero network dependency, ensuring privacy, speed, and reliability.

> 🔒 **Privacy First**: All inference happens locally on-device  
> ⚡ **High Performance**: Optimized for Apple Silicon and Intel Macs  
> 🎯 **Production Ready**: Built for real-world iOS and macOS applications

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Your iOS/macOS App                         │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Kuzco.shared                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │   predict()     │  │   instance()    │  │ Cache Management    │  │
│  │   ↓             │  │   ↓             │  │ ↓                   │  │
│  │ • Dialogue      │  │ • ModelProfile  │  │ • clearCache()      │  │
│  │ • Streaming     │  │ • Settings      │  │ • isModelCached()   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────┬───────────────────┬───────────────────────────┘
                      │                   │
              ┌───────▼──────┐   ┌────────▼────────┐
              │ Memory Cache │   │   ModelCache    │
              │  (Runtime)   │   │  (Persistent)   │
              │              │   │                 │
              │              │   │                 │ 
              │ Active       │   │ • Disk Storage  │
              │ Instances    │   │ • LRU Eviction  │
              │              │   │ • Validation    │
              └──────┬───────┘   └────────┬────────┘
                     │                    │
                     ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          LlamaInstance                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Model Loading   │  │ Context Mgmt    │  │ Token Generation    │  │
│  │ • Startup       │  │ • KV Cache      │  │ • Streaming         │  │
│  │ • Pre-warming   │  │ • Batching      │  │ • Stop Sequences    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         LlamaKitBridge                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Model Ops       │  │ Context Ops     │  │ Token Ops           │  │
│  │ • Load/Free     │  │ • Create/Free   │  │ • Tokenize          │  │
│  │ • Memory Map    │  │ • Batch Proc    │  │ • Detokenize        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           llama.cpp                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ GGUF Parser     │  │ Inference Eng   │  │ Hardware Accel      │  │
│  │ • Model Format  │  │ • Transformer   │  │ • Metal (GPU)       │  │
│  │ • Quantization  │  │ • Attention     │  │ • CPU Vectorization │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            File System                              │
│                                                                     │
│  Your App Bundle/Documents     App Support (Cache)                  │
│  ├── model1.gguf (4.2GB)       ├── KuzcoModelCache/                 │
│  ├── model2.gguf (7.1GB)       │   ├── cached_model_1.kuzco_cache   │
│  └── model3.gguf (2.8GB)       │   ├── cached_model_2.kuzco_cache   │
│                                │   └── kuzco_cache_metadata.json    │
└─────────────────────────────────────────────────────────────────────┘

📊 Data Flow:
┌─────┐    ┌─────────┐    ┌──────────┐    ┌─────────────┐    ┌─────────┐
│User │───▶│ Kuzco   │───▶│ Instance │───▶│ Bridge      │───▶│llama.cpp│
│Call │    │ Manager │    │ (Cached) │    │ (C++ Wrap)  │    │ Engine  │
└─────┘    └─────────┘    └──────────┘    └─────────────┘    └─────────┘
              │ ▲
              ▼ │ 🔄 Cache Hit: ~1-3 seconds
        ┌─────────────┐ 
        │ ModelCache  │ 
        │ (Persistent)│ 
        └─────────────┘ 
              │
              ▼ 💾 Cache Miss: Load from .gguf (~10-30 seconds)
        ┌─────────────┐
        │ Disk Storage│
        │(.gguf files)│
        └─────────────┘
```

## ✨ Key Features

### 🚀 **Core Capabilities**
- **Local LLM Execution**: Run powerful language models entirely on-device using `llama.cpp`
- **Multiple Model Architectures**: Support for LLaMA, Mistral, Phi, Gemma, and OpenChat models
- **Async/Await Native**: Modern Swift concurrency with streaming responses
- **Cross-Platform**: Works seamlessly on iOS, macOS, and Mac Catalyst

### ⚙️ **Advanced Configuration**
- **Flexible Model Settings**: Fine-tune context length, batch size, GPU layers, and CPU threads
- **Customizable Sampling**: Control temperature, top-K, top-P, repetition penalties, and Mirostat
- **Smart Resource Management**: Efficient instance caching and automatic context handling
- **Pre-warming Engine**: Optional model preloading for faster inference

### 🎨 **Developer Experience**
- **Simple API**: Get started with just a few lines of code
- **Comprehensive Error Handling**: Detailed error messages and recovery suggestions
- **Memory Efficient**: Optimized for mobile device constraints
- **Thread Safe**: Concurrent prediction support

## 📋 Requirements

- **iOS**: 15.0+
- **macOS**: 12.0+
- **Mac Catalyst**: 15.0+
- **Swift**: 5.9+
- **Xcode**: 15.0+

## 📦 Installation

### Swift Package Manager (Recommended)

#### Via Xcode
1. Open your project in Xcode
2. Go to **File** → **Add Package Dependencies**
3. Enter the repository URL:
   ```
   https://github.com/jcassoutt/Kuzco.git
   ```
4. Select **Up to Next Major Version** and click **Add Package**
5. Add `Kuzco` to your target

#### Via Package.swift
Add Kuzco to your `Package.swift` dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/jcassoutt/Kuzco.git", from: "1.0.0")
]
```

Then add it to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: ["Kuzco"]
)
```

## 🚀 Quick Start

### Basic Usage

```swift
import Kuzco

class ChatService {
    private let kuzco = Kuzco.shared
    
    func generateResponse(to userMessage: String) async throws {
        // 1. Create a model profile
        let profile = ModelProfile(
            sourcePath: "/path/to/your/model.gguf",
            architecture: .llama3
        )
        
        // 2. Prepare the conversation
        let dialogue: [Turn] = [
            Turn(role: .user, text: userMessage)
        ]
        
        // 3. Generate response with streaming
        let stream = try await kuzco.predict(
            dialogue: dialogue,
            with: profile,
            instanceSettings: .performanceFocused,
            predictionConfig: .creative
        )
        
        // 4. Process the streaming response
        for try await token in stream {
            print(token, terminator: "")
        }
        print() // New line after completion
    }
}
```

### Advanced Configuration

```swift
// Custom instance settings for better performance
let customSettings = InstanceSettings(
    contextLength: 4096,
    batchSize: 512,
    gpuOffloadLayers: 35,
    cpuThreads: 8,
    useFlashAttention: true
)

// Fine-tuned prediction config
let customConfig = PredictionConfig(
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    repeatPenalty: 1.1,
    maxTokens: 1024
)

// Use custom configurations
let stream = try await kuzco.predict(
    dialogue: dialogue,
    with: profile,
    instanceSettings: customSettings,
    predictionConfig: customConfig
)
```

### Multi-turn Conversation

```swift
var conversation: [Turn] = []

// Add user message
conversation.append(Turn(role: .user, text: "What is machine learning?"))

// Generate and collect AI response
var aiResponse = ""
let stream = try await kuzco.predict(
    dialogue: conversation,
    with: profile
)

for try await token in stream {
    aiResponse += token
}

// Add AI response to conversation history
conversation.append(Turn(role: .assistant, text: aiResponse))

// Continue the conversation...
conversation.append(Turn(role: .user, text: "Can you give me an example?"))
```

## 🧠 Supported Model Architectures

Kuzco supports multiple popular LLM architectures with optimized prompt formatting:

| Architecture | Models | Prompt Format |
|-------------|---------|---------------|
| **LLaMA 3** | Llama 3, Llama 3.1, Llama 3.2 | Optimized Llama 3 format |
| **LLaMA General** | Llama 2, Code Llama | Standard Llama format |
| **Mistral Instruct** | Mistral 7B, Mixtral 8x7B | Mistral chat format |
| **Phi** | Phi-3, Phi-3.5 | Microsoft Phi format |
| **Gemma** | Gemma 2B, Gemma 7B | Google Gemma format |
| **OpenChat** | OpenChat models | OpenChat conversation format |

### Adding Your Model

```swift
// For a LLaMA 3 model
let profile = ModelProfile(
    sourcePath: "/path/to/llama3-8b.gguf",
    architecture: .llama3
)

// For a Mistral model
let profile = ModelProfile(
    sourcePath: "/path/to/mistral-7b-instruct.gguf",
    architecture: .mistralInstruct
)
```

## ⚙️ Configuration Reference

### InstanceSettings

Controls how the model is loaded and executed:

```swift
let settings = InstanceSettings(
    contextLength: 4096,        // Context window size (tokens)
    batchSize: 512,            // Batch size for processing
    gpuOffloadLayers: 35,      // Layers to offload to GPU (Metal)
    cpuThreads: 8,             // CPU threads to use
    useFlashAttention: true,   // Enable flash attention optimization
    enableMemoryMapping: true  // Use memory mapping for efficiency
)
```

### PredictionConfig

Fine-tune the text generation behavior:

```swift
let config = PredictionConfig(
    temperature: 0.7,          // Randomness (0.0 = deterministic, 1.0+ = creative)
    topK: 40,                 // Top-K sampling
    topP: 0.9,                // Nucleus sampling
    repeatPenalty: 1.1,       // Repetition penalty
    maxTokens: 1024,          // Maximum tokens to generate
    stopSequences: ["</s>"],  // Stop generation at these sequences
    enableMirostat: false,    // Enable Mirostat sampling
    mirostatTau: 5.0,        // Mirostat target entropy
    mirostatEta: 0.1         // Mirostat learning rate
)
```

## 📱 Platform Considerations

### iOS Optimization
- **Memory Management**: Models are automatically unloaded when memory pressure is detected
- **Background Handling**: Inference pauses when app enters background
- **Battery Optimization**: Smart CPU/GPU usage based on device capabilities

### macOS Features
- **Apple Silicon**: Full Metal GPU acceleration on M1/M2/M3 Macs
- **Intel Macs**: Optimized CPU inference with vectorization
- **Memory Mapping**: Efficient large model loading

## 🔧 Troubleshooting

### Common Issues

**Q: My model isn't loading / crashes on load**
- Ensure your `.gguf` model file is compatible with llama.cpp
- Check that the file path is correct and accessible
- Verify you have enough available RAM for the model

**Q: Inference is slow**
- Increase `gpuOffloadLayers` for Apple Silicon devices
- Reduce `contextLength` if you don't need large contexts
- Try `instanceSettings: .performanceFocused`

**Q: Getting memory warnings on iOS**
- Use smaller quantized models (Q4_0, Q4_1)
- Reduce `contextLength` and `batchSize`
- Consider `instanceSettings: .memoryEfficient`

### Performance Tips

1. **Model Selection**: Use appropriately sized models for your target devices
2. **Quantization**: Q4_0 and Q4_1 models offer good quality/size balance
3. **Context Management**: Only include necessary conversation history
4. **Pre-warming**: Use `kuzco.prewarm()` for faster first inference

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
git clone https://github.com/jcassoutt/Kuzco.git
cd Kuzco
swift build
swift test
```

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - The foundational C++ library that makes this possible
- **[Georgi Gerganov](https://github.com/ggerganov)** - Creator of llama.cpp
- **Open Source Community** - For making efficient on-device AI a reality

## 📚 Additional Resources

- 📖 **[API Documentation](https://jcassoutt.github.io/Kuzco/documentation/kuzco/)** - Complete API reference
- 🧪 **[Example Projects](https://github.com/jcassoutt/Kuzco/tree/main/Examples)** - Sample implementations
- 💬 **[Discussions](https://github.com/jcassoutt/Kuzco/discussions)** - Community Q&A and feature requests
- 🐛 **[Issues](https://github.com/jcassoutt/Kuzco/issues)** - Bug reports and feature requests

---

<div align="center">
  <strong>Built with ❤️ for the Swift community</strong><br>
  <sub>Made by <a href="https://github.com/jaredcassoutt">Jared Cassoutt</a></sub>
</div>
