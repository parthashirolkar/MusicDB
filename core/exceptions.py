"""Custom exceptions for MusicDB."""


class MusicDBError(Exception):
    """Base exception for MusicDB."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AudioProcessingError(MusicDBError):
    """Raised when audio processing fails."""

    pass


class EmbeddingError(MusicDBError):
    """Raised when embedding generation fails."""

    pass


class DatabaseError(MusicDBError):
    """Raised when database operations fail."""

    pass


class DownloadError(MusicDBError):
    """Raised when YouTube download fails."""

    pass


class ValidationError(MusicDBError):
    """Raised when validation fails."""

    pass


class ConfigurationError(MusicDBError):
    """Raised when configuration is invalid."""

    pass
