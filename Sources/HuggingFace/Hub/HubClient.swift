import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

#if canImport(Xet)
    import Xet
#endif

/// A Hugging Face Hub API client.
///
/// This client provides methods to interact with the Hugging Face Hub API,
/// allowing you to list and retrieve information about models, datasets, and spaces,
/// as well as manage repositories.
///
/// The client automatically detects authentication tokens from standard locations (in order of priority):
/// - `HF_TOKEN` environment variable
/// - `HUGGING_FACE_HUB_TOKEN` environment variable
/// - `HF_TOKEN_PATH` environment variable (path to token file)
/// - `HF_HOME/token` file
/// - `~/.cache/huggingface/token` file (standard HF CLI location)
/// - `~/.huggingface/token` file (fallback location)
///
/// The endpoint can be customized via the `HF_ENDPOINT` environment variable.
///
/// - SeeAlso: [Hub API Documentation](https://huggingface.co/docs/hub/api)
public final class HubClient: Sendable {
    /// The default host URL for the Hugging Face Hub API.
    public static let defaultHost = URL(string: "https://huggingface.co")!

    /// A default client instance with auto-detected host and bearer token.
    ///
    /// This client automatically detects the authentication token from environment variables
    /// or standard token file locations, and uses the endpoint specified by `HF_ENDPOINT`
    /// environment variable (defaults to https://huggingface.co).
    public static let `default` = HubClient()

    /// Indicates whether Xet acceleration is enabled for this client.
    public let isXetEnabled: Bool

    /// The underlying HTTP client.
    internal let httpClient: HTTPClient
    
    #if canImport(Xet)
        /// Xet client instance for connection reuse (created once during initialization)
        private let xetClient: XetHubClient?
    #endif

    /// The host URL for requests made by the client.
    public var host: URL {
        httpClient.host
    }

    /// The value for the `User-Agent` header sent in requests, if any.
    public var userAgent: String? {
        httpClient.userAgent
    }

    /// The underlying client session.
    var session: URLSession {
        httpClient.session
    }

    /// The Bearer token for authentication, if any.
    public var bearerToken: String? {
        get async {
            try? await httpClient.tokenProvider.getToken()
        }
    }

    /// Creates a client with auto-detected host and bearer token.
    ///
    /// This initializer automatically detects the authentication token from standard locations
    /// and uses the endpoint specified by the `HF_ENDPOINT` environment variable.
    ///
    /// - Parameters:
    ///   - session: The underlying client session. Defaults to `URLSession(configuration: .default)`.
    ///   - userAgent: The value for the `User-Agent` header sent in requests, if any. Defaults to `nil`.
    ///   - xetConfiguration: Configuration for Xet downloads. Pass `nil` to disable Xet acceleration.
    ///     Defaults to `.highPerformance()` if Xet is supported, `nil` otherwise.
    public convenience init(
        session: URLSession = URLSession(configuration: .default),
        userAgent: String? = nil,
        xetConfiguration: XetConfiguration? = HubClient.isXetSupported ? .highPerformance() : nil
    ) {
        self.init(
            session: session,
            host: Self.detectHost(),
            userAgent: userAgent,
            tokenProvider: .environment,
            xetConfiguration: xetConfiguration
        )
    }

    /// Creates a client with the specified session, host, user agent, and authentication token.
    ///
    /// - Parameters:
    ///   - session: The underlying client session. Defaults to `URLSession(configuration: .default)`.
    ///   - host: The host URL to use for requests.
    ///   - userAgent: The value for the `User-Agent` header sent in requests, if any. Defaults to `nil`.
    ///   - bearerToken: The Bearer token for authentication, if any. Defaults to `nil`.
    ///   - xetConfiguration: Configuration for Xet downloads. Pass `nil` to disable Xet acceleration.
    ///     Defaults to `.highPerformance()` if Xet is supported, `nil` otherwise.
    public convenience init(
        session: URLSession = URLSession(configuration: .default),
        host: URL,
        userAgent: String? = nil,
        bearerToken: String? = nil,
        xetConfiguration: XetConfiguration? = HubClient.isXetSupported ? .highPerformance() : nil
    ) {
        self.init(
            session: session,
            host: host,
            userAgent: userAgent,
            tokenProvider: bearerToken.map { .fixed(token: $0) } ?? .none,
            xetConfiguration: xetConfiguration
        )
    }

    /// Creates a client with the specified session, host, user agent, and token provider.
    ///
    /// - Parameters:
    ///   - session: The underlying client session. Defaults to `URLSession(configuration: .default)`.
    ///   - host: The host URL to use for requests.
    ///   - userAgent: The value for the `User-Agent` header sent in requests, if any. Defaults to `nil`.
    ///   - tokenProvider: The token provider for authentication.
    ///   - xetConfiguration: Configuration for Xet downloads. Pass `nil` to disable Xet acceleration.
    ///     Defaults to `.highPerformance()` if Xet is supported, `nil` otherwise.
    public init(
        session: URLSession = URLSession(configuration: .default),
        host: URL,
        userAgent: String? = nil,
        tokenProvider: TokenProvider,
        xetConfiguration: XetConfiguration? = HubClient.isXetSupported ? .highPerformance() : nil
    ) {
        self.isXetEnabled = xetConfiguration != nil && HubClient.isXetSupported
        self.httpClient = HTTPClient(
            host: host,
            userAgent: userAgent,
            tokenProvider: tokenProvider,
            session: session
        )
        
        #if canImport(Xet)
            if let config = xetConfiguration, self.isXetEnabled {
                // Create XetHubClient once during initialization
                self.xetClient = XetHubClient(
                    sessionConfiguration: session.configuration,
                    configuration: config
                )
            } else {
                self.xetClient = nil
            }
        #endif
    }

    // MARK: - Auto-detection

    /// Detects the Hugging Face Hub endpoint from environment variables.
    ///
    /// Checks the `HF_ENDPOINT` environment variable, defaulting to https://huggingface.co.
    ///
    /// - Returns: The detected or default endpoint URL.
    private static func detectHost() -> URL {
        if let endpoint = ProcessInfo.processInfo.environment["HF_ENDPOINT"],
            let url = URL(string: endpoint)
        {
            return url
        }
        return defaultHost
    }

    public static var isXetSupported: Bool {
        #if canImport(Xet)
            return true
        #else
            return false
        #endif
    }

    // MARK: - Xet Client

    #if canImport(Xet)
        /// Returns the Xet client for faster downloads.
        ///
        /// The client is created once during initialization and reused across downloads
        /// to enable connection pooling and avoid reinitialization overhead.
        ///
        /// - Returns: A Xet client instance.
        internal func getXetClient() throws -> XetHubClient {
            guard isXetEnabled, let client = xetClient else {
                throw HTTPClientError.requestError("Xet support is disabled for this client.")
            }
            return client
        }
        
        /// Gets or fetches a CAS JWT for the given refresh route.
        ///
        /// JWTs are cached by the XetHubClient to avoid redundant API calls.
        ///
        /// - Parameters:
        ///   - refreshRoute: The refresh route URL for fetching the JWT
        ///   - token: Optional authentication token
        /// - Returns: A CAS JWT info object
        internal func getCachedJwt(
            refreshRoute: String,
            token: String?
        ) async throws -> CasJwtInfo {
            let xetClient = try getXetClient()
            return try await xetClient.getCasJwt(refreshRoute: refreshRoute, token: token)
        }
    #endif
    }
