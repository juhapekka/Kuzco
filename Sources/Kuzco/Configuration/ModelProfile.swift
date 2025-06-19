//
//  ModelProfile.swift
//  Kuzco
//
//  Created by Jared Cassoutt on 5/29/25.
//

import Foundation

/// Specifies the type of LLM architecture for prompt formatting and behavior.
public enum ModelArchitecture: String, CaseIterable, Codable, Hashable {
    case llamaGeneral
    case llama3
    case mistralInstruct
    case phiGeneric
    case gemmaInstruct
    case openChat
    case qwen2
    case qwen3
    case codellama
    case deepseek
    case commandR
    case yi
    case mixtral
    case unknown
    
    /// Attempts to detect architecture from model name/path
    public static func detectFromPath(_ path: String) -> ModelArchitecture {
        let lowercasedPath = path.lowercased()
        
        if lowercasedPath.contains("qwen2") {
            return .qwen2
        } else if lowercasedPath.contains("qwen3") {
            return .qwen3
        } else if lowercasedPath.contains("llama-3") || lowercasedPath.contains("llama3") {
            return .llama3
        } else if lowercasedPath.contains("codellama") || lowercasedPath.contains("code-llama") {
            return .codellama
        } else if lowercasedPath.contains("mistral") {
            return .mistralInstruct
        } else if lowercasedPath.contains("gemma") {
            return .gemmaInstruct
        } else if lowercasedPath.contains("phi") {
            return .phiGeneric
        } else if lowercasedPath.contains("deepseek") {
            return .deepseek
        } else if lowercasedPath.contains("command-r") || lowercasedPath.contains("commandr") {
            return .commandR
        } else if lowercasedPath.contains("yi-") {
            return .yi
        } else if lowercasedPath.contains("mixtral") {
            return .mixtral
        } else if lowercasedPath.contains("openchat") {
            return .openChat
        } else if lowercasedPath.contains("llama") {
            return .llamaGeneral
        } else {
            return .unknown
        }
    }
    
    /// Returns a fallback architecture that's more likely to be supported by older llama.cpp versions
    public func getFallbackArchitecture() -> ModelArchitecture {
        switch self {
        case .qwen3:
            return .qwen2  // Fallback qwen3 -> qwen2
        case .qwen2:
            return .unknown  // Fallback qwen2 -> unknown (uses ChatML)
        case .unknown:
            return .llamaGeneral  // Last resort fallback
        default:
            return .unknown  // General fallback for any unsupported architecture
        }
    }
}

public struct ModelProfile: Hashable, Codable {
    public let id: String
    public let sourcePath: String
    public let architecture: ModelArchitecture

    public init(id: String? = nil, sourcePath: String, architecture: ModelArchitecture? = nil) {
        self.id = id ?? sourcePath
        self.sourcePath = sourcePath
        self.architecture = architecture ?? ModelArchitecture.detectFromPath(sourcePath)
    }
    
    /// Safer initialization that uses fallback architectures for better compatibility
    public static func createWithFallback(id: String? = nil, sourcePath: String, architecture: ModelArchitecture? = nil) -> ModelProfile {
        let detectedArch = architecture ?? ModelArchitecture.detectFromPath(sourcePath)
        
        // For known problematic architectures, use fallback immediately
        let safeArch: ModelArchitecture
        switch detectedArch {
        case .qwen3:
            // qwen3 is often unsupported by older llama.cpp versions, fall back to qwen2
            print("🦙 Kuzco - Using qwen2 architecture as fallback for qwen3 model for better compatibility")
            safeArch = .qwen2
        default:
            safeArch = detectedArch
        }
        
        return ModelProfile(id: id, sourcePath: sourcePath, architecture: safeArch)
    }
    
    /// Validates that the model file exists and appears to be a valid GGUF file
    public func validateModelFile() throws {
        guard FileManager.default.fileExists(atPath: sourcePath) else {
            throw KuzcoError.modelFileNotAccessible(path: sourcePath)
        }
        
        // Check if it's a GGUF file by reading the magic bytes
        guard let fileHandle = FileHandle(forReadingAtPath: sourcePath) else {
            throw KuzcoError.modelFileNotAccessible(path: sourcePath)
        }
        
        defer { fileHandle.closeFile() }
        
        let magicBytes = fileHandle.readData(ofLength: 4)
        let expectedMagic = "GGUF".data(using: .ascii)!
        
        if magicBytes != expectedMagic {
            throw KuzcoError.configurationInvalid(reason: "Model file does not appear to be a valid GGUF format. Expected magic bytes 'GGUF', but found different format.")
        }
    }
    
    /// Provides suggestions for common model architecture issues
    public func getArchitectureSuggestions() -> String {
        switch architecture {
        case .qwen2, .qwen3:
            return "For Qwen models, ensure you're using a recent version of llama.cpp (post-2024) that supports Qwen architectures. Consider using the 'from' dependency format in Package.swift instead of a fixed revision."
        case .unknown:
            return "Architecture could not be auto-detected. Try specifying the architecture explicitly when creating the ModelProfile, or ensure the model filename contains recognizable architecture indicators (e.g., 'qwen3', 'llama3', 'mistral')."
        default:
            return "This architecture should be supported by most recent versions of llama.cpp."
        }
    }
}
