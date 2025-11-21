import Foundation
import Testing
@testable import HuggingFace

#if canImport(Xet)
    import Xet
#endif

/// Integration tests that exercise the high-level download path (including Xet) against the live Hugging Face Hub.
///
/// These tests perform large network transfers, so they are opt-in. Set the environment variable
/// `HF_RUN_SPEED_TEST=1` before running `swift test` to enable them.
@Suite struct DownloadSpeedTests {
    @Test("Xet download speed")
    func xetDownloadSpeed() async throws {
        print("xetDownloadSpeed started")
        guard ProcessInfo.processInfo.environment["HF_RUN_SPEED_TEST"] == "1" else {
            Issue.record("Set HF_RUN_SPEED_TEST=1 to enable the download speed integration test.")
            return
        }

        guard HubClient.isXetSupported else {
            Issue.record("Xet acceleration is not supported on this platform.")
            return
        }

        let env = ProcessInfo.processInfo.environment
        let repoIdentifier = env["HF_SPEED_TEST_REPO"] ?? "Qwen/Qwen3-0.6B"
        let minSizeMB = Int(env["HF_SPEED_TEST_MIN_SIZE_MB"] ?? "") ?? 10

        guard let repoID = Repo.ID(rawValue: repoIdentifier) else {
            Issue.record("Invalid repo identifier: \(repoIdentifier)")
            return
        }

        print("repoID: \(repoID)")
        print("minSizeMB: \(minSizeMB)")
        print("xetConfiguration: \(XetConfiguration.highPerformance())")
        let tester = DownloadSpeedTester(
            repo: repoID,
            minSizeMB: minSizeMB,
            xetConfiguration: .highPerformance()
        )

        let result: DownloadSpeedTester.Result
        do {
            print("tester.run() started")
            result = try await tester.run()
        } catch DownloadSpeedTester.RunError.noEligibleFiles {
            Issue.record("No files of at least \(minSizeMB) MB were found in \(repoID).")
            return
        }

        #expect(result.totalBytes > 0, "Expected to download at least one file.")
        #expect(result.averageSpeedBytesPerSecond > 0, "Average speed should be greater than zero.")

        Issue.record(
            """
            Downloaded \(DownloadSpeedTester.formatBytes(Int64(result.totalBytes))) \
            in \(String(format: "%.2f", result.totalTime))s \
            (\(DownloadSpeedTester.formatBytes(Int64(result.averageSpeedBytesPerSecond)))/s average)
            """
        )
    }
}

private struct DownloadSpeedTester {
    enum RunError: Error {
        case noEligibleFiles
    }

    struct Result {
        let totalBytes: Int
        let totalTime: TimeInterval

        var averageSpeedBytesPerSecond: Double {
            guard totalTime > 0 else { return 0 }
            return Double(totalBytes) / totalTime
        }
    }

    let repo: Repo.ID
    let minSizeMB: Int

    let xetConfiguration: XetConfiguration?

    init(repo: Repo.ID, minSizeMB: Int, xetConfiguration: XetConfiguration?) {
        self.repo = repo
        self.minSizeMB = minSizeMB
        self.xetConfiguration = xetConfiguration
    }

    func run() async throws -> Result {
        print("run() started")
        let client = HubClient(xetConfiguration: xetConfiguration)
        print("client: \(client)")

        let files = try await client.listFiles(
            in: repo,
            kind: .model,
            revision: "main",
            recursive: true
        )

        let testFiles = Self.selectTestFiles(from: files, minSizeMB: minSizeMB)
        guard !testFiles.isEmpty else { throw RunError.noEligibleFiles }

        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("hf-speed-test-\(UUID().uuidString)")
        print("tempDir: \(tempDir)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        print("tempDir created")
        defer {
            print("tempDir cleanup")
            try? FileManager.default.removeItem(at: tempDir)
        }

        var totalBytes = 0
        var totalTime: TimeInterval = 0

        for file in testFiles {
            print("downloading file: \(file.path)")
            let destination = tempDir.appendingPathComponent(file.path)
            try? FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            let start = Date()
            _ = try await client.downloadFile(
                at: file.path,
                from: repo,
                to: destination,
                kind: .model,
                revision: "main"
            )
            let duration = Date().timeIntervalSince(start)

            totalBytes += file.size ?? 0
            totalTime += duration
        }

        print("run() completed")
        return Result(totalBytes: totalBytes, totalTime: totalTime)
    }

    static func selectTestFiles(from files: [Git.TreeEntry], minSizeMB: Int) -> [Git.TreeEntry] {
        let minSizeBytes = minSizeMB * 1_024 * 1_024
        let largeFiles = files.filter { file in
            file.type == .file && (file.size ?? 0) >= minSizeBytes
        }

        guard !largeFiles.isEmpty else { return [] }

        var selected: [Git.TreeEntry] = []
        let priorities = ["safetensors", "bin", "gguf", "pt", "pth"]

        for priority in priorities {
            if let match = largeFiles.first(where: { $0.path.contains(priority) }) {
                if !selected.contains(where: { $0.path == match.path }) {
                    selected.append(match)
                }
            }
        }

        if selected.count < 3 {
            let remaining = largeFiles.filter { candidate in
                !selected.contains(where: { $0.path == candidate.path })
            }
            selected.append(contentsOf: remaining.sorted { ($0.size ?? 0) > ($1.size ?? 0) })
        }

        return Array(selected.prefix(3))
    }

    static func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useKB, .useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}
