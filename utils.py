"""Utility functions for MusicDB."""

import re
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    """Sanitize a string for use as filename.

    Args:
        filename: Input string

    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized.strip()


def format_duration(seconds: float | None) -> str:
    """Format duration in seconds to human readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "3:45"
    """
    if seconds is None:
        return "unknown"

    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def format_number(num: int | None) -> str:
    """Format large numbers with K/M suffixes.

    Args:
        num: Number to format

    Returns:
        Formatted string
    """
    if num is None:
        return "unknown"

    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max length.

    Args:
        text: Input text
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def find_audio_files(
    directory: Path | str, extensions: list[str] | None = None
) -> list[Path]:
    """Find all audio files in directory.

    Args:
        directory: Directory to search
        extensions: List of extensions to include

    Returns:
        List of file paths
    """
    directory = Path(directory)

    if extensions is None:
        extensions = [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"]

    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(files)
