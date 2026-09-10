"""Query database for similar songs."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from services.chroma_service import ChromaService
from services.mert_service import MERTService
from services.audio_service import AudioService
from services.feature_service import FeatureService
from services.reranker_service import AudioFeatureReranker
from core.config import get_settings
from utils import format_duration


def search_similar_songs(
    query_path: Path | str, n_results: int = 5, collection_name: str | None = None
) -> list[dict]:
    """Search for songs similar to the query audio.

    Args:
        query_path: Path to query audio file
        n_results: Number of results to return
        collection_name: Optional collection override

    Returns:
        List of result dictionaries
    """
    query_path = Path(query_path)

    if not query_path.exists():
        logger.error(f"Query file not found: {query_path}")
        return []

    # Initialize services
    db = ChromaService(collection_name=collection_name)
    embedder = MERTService()
    settings = get_settings()

    try:
        # Generate query embedding
        logger.info(f"Processing query: {query_path.name}")
        query_embedding = embedder.embed_file(query_path)

        overfetch = settings.reranker.overfetch if settings.reranker.enabled else 0
        results = db.search_similar(
            query_embedding.tolist(), n_results=n_results + overfetch
        )

        if settings.reranker.enabled and results:
            audio_svc = AudioService()
            feature_svc = FeatureService()
            reranker = AudioFeatureReranker()

            query_y = audio_svc.load_audio(query_path)
            query_features = feature_svc.extract_features(
                query_y, audio_svc.sample_rate
            )
            results = reranker.rerank(query_features, results, top_n=n_results)

        return results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def print_results(results: list[dict]) -> None:
    """Pretty print search results.

    Args:
        results: List of result dicts
    """
    if not results:
        print("\nNo similar songs found.")
        return

    print(f"\n🎵 Found {len(results)} similar songs:\n")

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        filename = metadata.get("filename", "Unknown")
        duration = metadata.get("duration_seconds")
        similarity = result.get("reranked_score", 1 - result["distance"])

        print(f"{i}. {filename}")
        print(f"   Similarity: {similarity:.3f}")
        print(f"   Duration: {format_duration(duration)}")
        print()


def interactive_search(music_dir: Path | str = Path("inference_music_files")) -> None:
    """Interactive search mode.

    Args:
        music_dir: Directory with query files
    """
    music_dir = Path(music_dir)

    if not music_dir.exists():
        logger.error(f"Directory not found: {music_dir}")
        return

    # Find audio files
    files = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))

    if not files:
        logger.error(f"No audio files in {music_dir}")
        return

    print(f"\nFound {len(files)} files. Using: {files[-1].name}\n")

    # Search
    results = search_similar_songs(files[-1])
    print_results(results)


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    # Run interactive search
    interactive_search()
