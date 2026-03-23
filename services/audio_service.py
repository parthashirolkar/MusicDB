"""Audio processing service with chunking."""

import numpy as np
import librosa
from pathlib import Path
from loguru import logger

from core.config import get_settings
from core.exceptions import AudioProcessingError


class AudioService:
    """Service for audio loading and preprocessing."""

    def __init__(self, sample_rate: int | None = None):
        """Initialize audio service.

        Args:
            sample_rate: Target sample rate (defaults to MERT's 24kHz)
        """
        settings = get_settings()
        self.sample_rate = sample_rate or settings.mert.sample_rate
        self.chunk_duration = settings.processing.chunk_duration
        self.chunk_overlap = settings.processing.chunk_overlap

    def load_audio(self, file_path: Path | str) -> np.ndarray:
        """Load and resample audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Audio waveform as numpy array

        Raises:
            AudioProcessingError: If loading fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise AudioProcessingError(f"File not found: {file_path}")

        try:
            logger.debug(f"Loading audio: {file_path}")
            audio, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=True)

            if len(audio) == 0:
                raise AudioProcessingError(f"Empty audio file: {file_path}")

            logger.debug(f"Loaded audio: {len(audio)} samples at {self.sample_rate}Hz")
            return audio

        except Exception as e:
            raise AudioProcessingError(f"Failed to load {file_path}: {e}")

    def chunk_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split audio into overlapping chunks.

        Args:
            audio: Audio waveform

        Returns:
            List of audio chunks
        """
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        overlap_samples = int(self.chunk_overlap * self.sample_rate)
        hop_samples = chunk_samples - overlap_samples

        chunks = []
        start = 0

        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]

            # Pad last chunk if necessary
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant")

            chunks.append(chunk)
            start += hop_samples

        logger.debug(f"Split audio into {len(chunks)} chunks")
        return chunks

    def validate_audio_file(self, file_path: Path | str) -> dict:
        """Validate audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Validation result dict
        """
        file_path = Path(file_path)
        result = {"valid": True, "errors": [], "warnings": []}

        try:
            info = librosa.get_duration(path=str(file_path))
            settings = get_settings()

            if info > settings.processing.max_duration:
                result["warnings"].append(
                    f"Audio longer than {settings.processing.max_duration}s, will be truncated"
                )

            if info < 1.0:
                result["errors"].append("Audio too short (< 1 second)")
                result["valid"] = False

        except Exception as e:
            result["errors"].append(f"Cannot read audio: {e}")
            result["valid"] = False

        return result

    def get_duration(self, file_path: Path | str) -> float:
        """Get audio duration in seconds.

        Args:
            file_path: Path to audio file

        Returns:
            Duration in seconds
        """
        return librosa.get_duration(path=str(file_path))
