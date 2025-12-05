import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// A cache manager for Hugging Face Hub files.
///
/// `HubCache` implements a Python-compatible cache structure
/// that allows sharing cached files between Swift and Python Hugging Face clients.
/// The cache uses content-addressed storage with symlinks
/// to efficiently store and share files across multiple revisions.
///
/// ## Cache Structure
///
/// The cache follows the Python `huggingface_hub` layout:
///
/// ```
/// ~/.cache/huggingface/hub/
/// ├── models--<namespace>--<repo>/
/// │   ├── blobs/
/// │   │   └── <etag>           # actual file content
/// │   ├── refs/
/// │   │   └── main             # contains commit hash
/// │   └── snapshots/
/// │       └── <commit_hash>/
/// │           └── <filename>   # symlink → ../../blobs/<etag>
/// ```
///
/// ## Cache Location
///
/// The cache location is determined by `CacheLocationProvider`, which by default
/// checks in the following order:
/// 1. `HF_HUB_CACHE` environment variable
/// 2. `HF_HOME` environment variable + `/hub`
/// 3. `~/.cache/huggingface/hub` (default)
///
/// ## Usage
///
/// ```swift
/// // Use default cache (auto-detected location)
/// let cache = HubCache.default
///
/// // Use a custom cache location
/// let customCache = HubCache(location: .fixed(directory: myCustomURL))
///
/// // Or with a path string
/// let pathCache = HubCache(location: .init(path: "~/my-hf-cache"))
///
/// // Check if a file is cached
/// if let cachedPath = cache.cachedFilePath(
///     repo: "openai-community/gpt2",
///     kind: .model,
///     revision: "main",
///     filename: "config.json"
/// ) {
///     let data = try Data(contentsOf: cachedPath)
/// }
///
/// // Store a file in cache
/// try cache.storeFile(
///     at: downloadedFileURL,
///     repo: "openai-community/gpt2",
///     kind: .model,
///     revision: "abc123...",
///     filename: "config.json",
///     etag: "abc123def456"
/// )
/// ```
public struct HubCache: Sendable {
    /// The default cache instance using auto-detected cache directory.
    public static let `default` = HubCache()

    /// The root directory of the cache.
    public let cacheDirectory: URL

    /// The location provider used to determine the cache directory.
    public let locationProvider: CacheLocationProvider

    /// Creates a cache manager with the specified location provider.
    ///
    /// - Parameter location: The location provider for determining the cache directory.
    ///   Defaults to `.environment` for auto-detection from environment variables.
    public init(location: CacheLocationProvider = .environment) {
        self.locationProvider = location
        self.cacheDirectory = location.resolve() ?? Self.fallbackDirectory
    }

    /// Creates a cache manager with the specified cache directory.
    ///
    /// - Parameter cacheDirectory: The root directory for the cache.
    public init(cacheDirectory: URL) {
        self.locationProvider = .fixed(directory: cacheDirectory)
        self.cacheDirectory = cacheDirectory
    }

    /// The fallback cache directory when resolution fails.
    private static var fallbackDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
    }

    // MARK: - Repository Path

    /// Returns the cache directory for a specific repository.
    ///
    /// The directory name follows the Python format: `<kind>--<namespace>--<repo>`
    /// For example: `models--openai-community--gpt2`
    ///
    /// - Parameters:
    ///   - repo: The repository identifier (e.g., "openai-community/gpt2").
    ///   - kind: The kind of repository.
    /// - Returns: The URL to the repository's cache directory.
    public func repoDirectory(repo: Repo.ID, kind: Repo.Kind) -> URL {
        let repoName = repo.description.replacingOccurrences(of: "/", with: "--")
        let dirName = "\(kind.pluralized)--\(repoName)"
        return cacheDirectory.appendingPathComponent(dirName)
    }

    /// Returns the blobs directory for a repository.
    public func blobsDirectory(repo: Repo.ID, kind: Repo.Kind) -> URL {
        repoDirectory(repo: repo, kind: kind).appendingPathComponent("blobs")
    }

    /// Returns the refs directory for a repository.
    public func refsDirectory(repo: Repo.ID, kind: Repo.Kind) -> URL {
        repoDirectory(repo: repo, kind: kind).appendingPathComponent("refs")
    }

    /// Returns the snapshots directory for a repository.
    public func snapshotsDirectory(repo: Repo.ID, kind: Repo.Kind) -> URL {
        repoDirectory(repo: repo, kind: kind).appendingPathComponent("snapshots")
    }

    // MARK: - Revision Resolution

    /// Resolves a reference (branch/tag) to its commit hash.
    ///
    /// Reads the refs file to get the commit hash for a given reference.
    ///
    /// - Parameters:
    ///   - repo: The repository identifier.
    ///   - kind: The kind of repository.
    ///   - ref: The reference name (e.g., "main").
    /// - Returns: The commit hash if found, `nil` otherwise.
    public func resolveRevision(repo: Repo.ID, kind: Repo.Kind, ref: String) -> String? {
        let refFile = refsDirectory(repo: repo, kind: kind).appendingPathComponent(ref)
        guard let contents = try? String(contentsOf: refFile, encoding: .utf8) else {
            return nil
        }
        return contents.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Updates the reference file with a new commit hash.
    ///
    /// - Parameters:
    ///   - repo: The repository identifier.
    ///   - kind: The kind of repository.
    ///   - ref: The reference name (e.g., "main", "refs/pr/5").
    ///   - commit: The commit hash to store.
    public func updateRef(repo: Repo.ID, kind: Repo.Kind, ref: String, commit: String) throws {
        let refsDir = refsDirectory(repo: repo, kind: kind)
        let refFile = refsDir.appendingPathComponent(ref)

        // Create parent directories for nested refs (e.g., "refs/pr/5")
        try FileManager.default.createDirectory(
            at: refFile.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        try commit.write(to: refFile, atomically: true, encoding: .utf8)
    }

    // MARK: - Cache Lookup

    /// Returns the path to a cached file if it exists.
    ///
    /// This method checks if a file is cached for the given repository and revision.
    /// It first resolves the revision (if it's a branch name) to a commit hash,
    /// then looks for the file in the snapshots directory.
    ///
    /// - Parameters:
    ///   - repo: The repository identifier.
    ///   - kind: The kind of repository.
    ///   - revision: The revision (commit hash or branch name).
    ///   - filename: The filename within the repository.
    /// - Returns: The URL to the cached file if found, `nil` otherwise.
    public func cachedFilePath(
        repo: Repo.ID,
        kind: Repo.Kind,
        revision: String,
        filename: String
    ) -> URL? {
        // Try to resolve the revision if it looks like a branch name
        let commitHash: String
        if isCommitHash(revision) {
            commitHash = revision
        } else if let resolved = resolveRevision(repo: repo, kind: kind, ref: revision) {
            commitHash = resolved
        } else {
            // Can't resolve revision, not cached
            return nil
        }

        let snapshotFile = snapshotsDirectory(repo: repo, kind: kind)
            .appendingPathComponent(commitHash)
            .appendingPathComponent(filename)

        // Check if the snapshot file exists (could be symlink or regular file)
        if FileManager.default.fileExists(atPath: snapshotFile.path) {
            return snapshotFile
        }

        return nil
    }

    /// Returns the path to a cached blob by its etag.
    ///
    /// - Parameters:
    ///   - repo: The repository identifier.
    ///   - kind: The kind of repository.
    ///   - etag: The normalized etag of the blob.
    /// - Returns: The URL to the blob if it exists, `nil` otherwise.
    public func cachedBlobPath(repo: Repo.ID, kind: Repo.Kind, etag: String) -> URL? {
        let normalizedEtag = normalizeEtag(etag)
        let blobPath = blobsDirectory(repo: repo, kind: kind)
            .appendingPathComponent(normalizedEtag)

        if FileManager.default.fileExists(atPath: blobPath.path) {
            return blobPath
        }

        return nil
    }

    // MARK: - File Storage

    /// Stores a file in the cache.
    ///
    /// This method:
    /// 1. Copies the file to the blobs directory (content-addressed by etag)
    /// 2. Creates a symlink in the snapshots directory pointing to the blob
    /// 3. Updates the refs file if the revision is a branch name
    ///
    /// On platforms that don't support symlinks (like some Windows configurations),
    /// the file is copied directly to the snapshots directory instead.
    ///
    /// File locking is used to coordinate concurrent access to blob files,
    /// preventing race conditions when multiple processes download the same file.
    ///
    /// - Parameters:
    ///   - sourceURL: The URL of the file to store.
    ///   - repo: The repository identifier.
    ///   - kind: The kind of repository.
    ///   - revision: The commit hash for this file.
    ///   - filename: The filename within the repository.
    ///   - etag: The etag of the file (used for content addressing).
    ///   - ref: Optional reference name to update (e.g., "main").
    public func storeFile(
        at sourceURL: URL,
        repo: Repo.ID,
        kind: Repo.Kind,
        revision: String,
        filename: String,
        etag: String,
        ref: String? = nil
    ) throws {
        let normalizedEtag = normalizeEtag(etag)

        // Create directories
        let blobsDir = blobsDirectory(repo: repo, kind: kind)
        let snapshotsDir = snapshotsDirectory(repo: repo, kind: kind)
            .appendingPathComponent(revision)

        try FileManager.default.createDirectory(at: blobsDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: snapshotsDir, withIntermediateDirectories: true)

        // Store blob (content-addressed) with file locking
        let blobPath = blobsDir.appendingPathComponent(normalizedEtag)
        let lock = FileLock(path: blobPath)

        try lock.withLock {
            if !FileManager.default.fileExists(atPath: blobPath.path) {
                try FileManager.default.copyItem(at: sourceURL, to: blobPath)
            }
        }

        // Create snapshot entry (symlink or copy)
        let snapshotPath = snapshotsDir.appendingPathComponent(filename)

        // Create parent directories for nested filenames
        let snapshotParent = snapshotPath.deletingLastPathComponent()
        if snapshotParent != snapshotsDir {
            try FileManager.default.createDirectory(
                at: snapshotParent,
                withIntermediateDirectories: true
            )
        }

        // Remove existing snapshot entry if present
        try? FileManager.default.removeItem(at: snapshotPath)

        // Try to create symlink, fall back to copy
        let relativeBlobPath = relativePathToBlob(from: snapshotPath, blobName: normalizedEtag)
        if !createSymlink(at: snapshotPath, pointingTo: relativeBlobPath) {
            // Symlinks not supported, copy the file instead
            try FileManager.default.copyItem(at: blobPath, to: snapshotPath)
        }

        // Update ref if provided
        if let ref = ref {
            try updateRef(repo: repo, kind: kind, ref: ref, commit: revision)
        }
    }

    /// Stores data directly in the cache.
    ///
    /// File locking is used to coordinate concurrent access to blob files,
    /// preventing race conditions when multiple processes download the same file.
    ///
    /// - Parameters:
    ///   - data: The data to store.
    ///   - repo: The repository identifier.
    ///   - kind: The kind of repository.
    ///   - revision: The commit hash for this file.
    ///   - filename: The filename within the repository.
    ///   - etag: The etag of the file.
    ///   - ref: Optional reference name to update.
    public func storeData(
        _ data: Data,
        repo: Repo.ID,
        kind: Repo.Kind,
        revision: String,
        filename: String,
        etag: String,
        ref: String? = nil
    ) throws {
        let normalizedEtag = normalizeEtag(etag)

        // Create directories
        let blobsDir = blobsDirectory(repo: repo, kind: kind)
        let snapshotsDir = snapshotsDirectory(repo: repo, kind: kind)
            .appendingPathComponent(revision)

        try FileManager.default.createDirectory(at: blobsDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: snapshotsDir, withIntermediateDirectories: true)

        // Store blob with file locking
        let blobPath = blobsDir.appendingPathComponent(normalizedEtag)
        let lock = FileLock(path: blobPath)

        try lock.withLock {
            if !FileManager.default.fileExists(atPath: blobPath.path) {
                try data.write(to: blobPath, options: .atomic)
            }
        }

        // Create snapshot entry
        let snapshotPath = snapshotsDir.appendingPathComponent(filename)

        // Create parent directories for nested filenames
        let snapshotParent = snapshotPath.deletingLastPathComponent()
        if snapshotParent != snapshotsDir {
            try FileManager.default.createDirectory(
                at: snapshotParent,
                withIntermediateDirectories: true
            )
        }

        // Remove existing snapshot entry if present
        try? FileManager.default.removeItem(at: snapshotPath)

        // Try to create symlink, fall back to copy
        let relativeBlobPath = relativePathToBlob(from: snapshotPath, blobName: normalizedEtag)
        if !createSymlink(at: snapshotPath, pointingTo: relativeBlobPath) {
            try data.write(to: snapshotPath, options: .atomic)
        }

        // Update ref if provided
        if let ref = ref {
            try updateRef(repo: repo, kind: kind, ref: ref, commit: revision)
        }
    }

    // MARK: -

    /// Normalizes an etag by removing quotes and the "W/" prefix.
    ///
    /// Python's huggingface_hub strips these when using etags as blob names.
    ///
    /// - Parameter etag: The raw etag from the HTTP response.
    /// - Returns: The normalized etag suitable for use as a filename.
    public func normalizeEtag(_ etag: String) -> String {
        var normalized = etag

        // Remove weak validator prefix
        if normalized.hasPrefix("W/") {
            normalized = String(normalized.dropFirst(2))
        }

        // Remove surrounding quotes
        normalized = normalized.trimmingCharacters(in: CharacterSet(charactersIn: "\""))

        return normalized
    }

    /// Checks if a string looks like a commit hash (40 hex characters).
    private func isCommitHash(_ string: String) -> Bool {
        guard string.count == 40 else { return false }
        return string.allSatisfy { $0.isHexDigit }
    }

    /// Calculates the relative path from a snapshot file to its blob.
    ///
    /// The path needs to go up to the repo directory, then into blobs.
    /// For a file at `snapshots/<commit>/<filename>`, the path is `../../blobs/<etag>`.
    /// For nested files like `snapshots/<commit>/path/to/file`, we need more `..`.
    private func relativePathToBlob(from snapshotPath: URL, blobName: String) -> String {
        // Count directory depth from snapshots/<commit>/ to the file
        let snapshotComponents = snapshotPath.pathComponents
        guard let snapshotsIndex = snapshotComponents.lastIndex(of: "snapshots") else {
            return "../../blobs/\(blobName)"
        }

        // Components after "snapshots": commit hash + filename path components
        let componentsAfterSnapshots = snapshotComponents.count - snapshotsIndex - 1
        // We need to go up (componentsAfterSnapshots) levels to reach the repo root
        let upPath = String(repeating: "../", count: componentsAfterSnapshots)

        return "\(upPath)blobs/\(blobName)"
    }

    /// Attempts to create a symbolic link.
    ///
    /// - Parameters:
    ///   - path: The path where the symlink should be created.
    ///   - destination: The destination path the symlink points to.
    /// - Returns: `true` if the symlink was created successfully, `false` otherwise.
    private func createSymlink(at path: URL, pointingTo destination: String) -> Bool {
        do {
            try FileManager.default.createSymbolicLink(
                atPath: path.path,
                withDestinationPath: destination
            )
            return true
        } catch {
            // Symlinks not supported on this platform/configuration
            return false
        }
    }
}
