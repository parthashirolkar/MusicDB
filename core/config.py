"""Configuration management using Pydantic."""

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """ChromaDB settings."""

    model_config = SettingsConfigDict(env_prefix="CHROMA_DB_")

    path: Path = Field(default=Path("./chroma_data"))
    collection_name: str = Field(default="song_vector_collection")

    @field_validator("path")
    @classmethod
    def ensure_path_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


class MERTSettings(BaseSettings):
    """MERT model settings."""

    model_config = SettingsConfigDict(env_prefix="MERT_")

    model_name: str = Field(default="m-a-p/MERT-v1-95M")
    sample_rate: int = Field(default=24000)
    embedding_dim: int = Field(default=768)

    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v: str) -> str:
        valid = ["m-a-p/MERT-v1-95M", "m-a-p/MERT-v1-330M"]
        if v not in valid:
            raise ValueError(f"Model must be one of: {valid}")
        return v


class ProcessingSettings(BaseSettings):
    """Audio processing settings."""

    model_config = SettingsConfigDict(env_prefix="MUSICDB_")

    validate_audio: bool = Field(default=True)
    normalize_embeddings: bool = Field(default=True)
    chunk_duration: int = Field(default=10, ge=1)
    chunk_overlap: int = Field(default=2, ge=0)
    max_duration: float = Field(default=600.0, ge=1)
    enable_gpu: bool = Field(default=True)

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        if "chunk_duration" in info.data and v >= info.data["chunk_duration"]:
            raise ValueError("chunk_overlap must be less than chunk_duration")
        return v


class DownloadSettings(BaseSettings):
    """YouTube download settings."""

    model_config = SettingsConfigDict(env_prefix="MUSICDB_")

    output_dir: Path = Field(default=Path("music_files"))
    audio_format: str = Field(default="mp3")
    audio_quality: str = Field(default="bestaudio")
    cleanup_files: bool = Field(default=False)

    @field_validator("output_dir")
    @classmethod
    def ensure_dir_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Allow extra env vars not mapped to fields
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    mert: MERTSettings = Field(default_factory=MERTSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    download: DownloadSettings = Field(default_factory=DownloadSettings)
    log_level: str = Field(default="INFO")


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
