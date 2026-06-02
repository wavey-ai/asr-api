// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "AsrMLXRuntime",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .library(name: "AsrMLXRuntime", targets: ["AsrMLXRuntime"]),
        .executable(name: "asr-mlx-transcribe", targets: ["AsrMLXTranscribe"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.3"),
    ],
    targets: [
        .target(
            name: "AsrMLXRuntime",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/AsrMLXRuntime"
        ),
        .executableTarget(
            name: "AsrMLXTranscribe",
            dependencies: ["AsrMLXRuntime"],
            path: "Sources/AsrMLXTranscribe"
        ),
    ]
)
