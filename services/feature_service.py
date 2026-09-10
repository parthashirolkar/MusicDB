"""Audio feature extraction service using librosa."""

from pathlib import Path

import librosa
import numpy as np


class FeatureService:
    """Service for extracting audio features for reranking."""

    def extract_features(
        self, y: np.ndarray, sr: int = 24000
    ) -> dict[str, float | list[float]]:
        """Extract audio features from a waveform.

        Args:
            y: Audio waveform array
            sr: Sample rate

        Returns:
            Dict of feature name to value (float or list[float] for chroma)
        """
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.ravel(tempo)[0]) if np.ndim(tempo) > 0 else float(tempo)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rms = librosa.feature.rms(y=y)[0]
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

        return {
            "tempo": tempo,
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "spectral_centroid_std": float(np.std(spectral_centroid)),
            "rms_energy_mean": float(np.mean(rms)),
            "chroma_mean": [float(v) for v in np.mean(chroma, axis=1)],
            "zero_crossing_rate_mean": float(np.mean(zcr)),
            "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        }

    def extract_features_from_file(
        self, filepath: Path | str
    ) -> dict[str, float | list[float]]:
        """Extract audio features from a file.

        Args:
            filepath: Path to audio file

        Returns:
            Dict of feature name to value
        """
        filepath = Path(filepath)
        y, sr = librosa.load(str(filepath), sr=24000, mono=True)
        return self.extract_features(y, sr)
