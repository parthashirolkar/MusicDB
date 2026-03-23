"""MERT embedding service."""

import torch
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from loguru import logger

from core.config import get_settings
from core.exceptions import EmbeddingError
from services.audio_service import AudioService


class MERTService:
    """Service for generating MERT embeddings."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize: bool = True,
    ):
        """Initialize MERT service.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
            normalize: Whether to L2-normalize embeddings
        """
        settings = get_settings()

        self.model_name = model_name or settings.mert.model_name
        self.device = self._get_device(device)
        self.normalize = normalize

        self._processor: Wav2Vec2FeatureExtractor | None = None
        self._model: AutoModel | None = None
        self._audio_service = AudioService()

        logger.info(
            f"MERTService initialized: model={self.model_name}, device={self.device}"
        )

    def _get_device(self, device: str | None) -> str:
        """Determine compute device."""
        if device:
            return device

        settings = get_settings()
        if settings.processing.enable_gpu and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self) -> None:
        """Lazy-load the MERT model."""
        if self._model is not None:
            return

        try:
            logger.info(f"Loading MERT model: {self.model_name}")

            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model.to(self.device)
            self._model.eval()

            logger.info("MERT model loaded successfully")

        except Exception as e:
            raise EmbeddingError(f"Failed to load MERT model: {e}")

    def generate_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Generate embedding from audio.

        Args:
            audio: Audio waveform at MERT sample rate

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        self._load_model()

        try:
            # Process audio
            inputs = self._processor(
                audio,
                sampling_rate=self._audio_service.sample_rate,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use mean of last hidden state as embedding
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            # Flatten to 1D
            embedding = embedding.flatten()

            # Normalize if requested
            if self.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            return embedding

        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}")

    def embed_file(
        self, file_path: Path | str, use_chunking: bool = True
    ) -> np.ndarray:
        """Generate embedding from audio file.

        Args:
            file_path: Path to audio file
            use_chunking: Whether to use chunking for long files

        Returns:
            Embedding vector
        """
        file_path = Path(file_path)

        # Load audio
        audio = self._audio_service.load_audio(file_path)

        if (
            use_chunking
            and len(audio)
            > self._audio_service.chunk_duration * self._audio_service.sample_rate
        ):
            # Use chunking for long files
            return self._embed_with_chunking(audio)
        else:
            # Single pass for short files
            return self.generate_embedding(audio)

    def _embed_with_chunking(self, audio: np.ndarray) -> np.ndarray:
        """Generate embedding using chunking and pooling.

        Args:
            audio: Full audio waveform

        Returns:
            Pooled embedding vector
        """
        chunks = self._audio_service.chunk_audio(audio)

        logger.debug(f"Embedding {len(chunks)} chunks")

        embeddings = []
        for i, chunk in enumerate(chunks):
            try:
                emb = self.generate_embedding(chunk)
                embeddings.append(emb)
                logger.debug(f"Embedded chunk {i + 1}/{len(chunks)}")
            except Exception as e:
                logger.warning(f"Failed to embed chunk {i + 1}: {e}")

        if not embeddings:
            raise EmbeddingError("No chunks could be embedded")

        # Pool embeddings by averaging
        pooled = np.mean(embeddings, axis=0)

        # Renormalize after pooling
        if self.normalize:
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm

        logger.debug(f"Pooled {len(embeddings)} embeddings")
        return pooled

    def embed_batch(self, file_paths: list[Path | str]) -> list[np.ndarray]:
        """Generate embeddings for multiple files.

        Args:
            file_paths: List of audio file paths

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for path in file_paths:
            try:
                emb = self.embed_file(path)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Failed to embed {path}: {e}")
                embeddings.append(None)

        return embeddings
