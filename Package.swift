// swift-tools-version:5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Kuzco",
    platforms: [
        .macOS(.v12),
        .iOS(.v15),
        .macCatalyst(.v15)
    ],
    products: [
        .library(
            name: "Kuzco",
            targets: ["Kuzco"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ggerganov/llama.cpp", revision: "9b75f03"),
    ],
    targets: [
        .target(
            name: "Kuzco",
            dependencies: [
                .product(name: "llama", package: "llama.cpp")
            ]),
        .testTarget(
            name: "KuzcoTests",
            dependencies: ["Kuzco"]),
    ]
)
