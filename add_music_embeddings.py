"""Add music embeddings using MERT model and ChromaDB."""

import sys
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from services.chroma_service import ChromaService
from services.mert_service import MERTService
from services.audio_service import AudioService
from utils import find_audio_files, sanitize_filename


def process_music_directory(
    music_dir: Path | str = Path("music_files"),
    collection_name: str | None = None,
) -> None:
    """Process all music files in a directory and add to database.

    Args:
        music_dir: Directory containing music files
        collection_name: Optional collection name override
    """
    # Initialize services
    db = ChromaService(collection_name=collection_name)
    embedder = MERTService()
    audio_svc = AudioService()

    # Find audio files
    music_dir = Path(music_dir)
    files = find_audio_files(music_dir)

    if not files:
        logger.warning(f"No audio files found in {music_dir}")
        return

    logger.info(f"Found {len(files)} audio files to process")

    # Process each file
    for i, file_path in enumerate(files, 1):
        logger.info(f"Processing {i}/{len(files)}: {file_path.name}")

        try:
            # Validate audio
            validation = audio_svc.validate_audio_file(file_path)
            if not validation["valid"]:
                logger.error(f"Validation failed: {validation['errors']}")
                continue

            # Generate embedding
            embedding = embedder.embed_file(file_path)
            logger.debug(f"Generated embedding: {embedding.shape}")

            # Create metadata
            song_id = sanitize_filename(file_path.stem)
            metadata = {
                "filename": file_path.name,
                "filepath": str(file_path),
                "duration_seconds": audio_svc.get_duration(file_path),
            }

            # Check for duplicates
            if db.song_exists(song_id):
                logger.warning(f"Song already exists, skipping: {song_id}")
                continue

            # Add to database
            db.add_song(song_id, embedding.tolist(), metadata)
            logger.success(f"Added: {song_id}")

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            continue

    # Summary
    total = db.count()
    logger.info(f"Database now contains {total} songs")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    # Run
    process_music_directory()
