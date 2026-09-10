"""Main music pipeline: Download from YouTube and add to database."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from services.chroma_service import ChromaService
from services.mert_service import MERTService
from services.youtube_service import YouTubeService
from services.audio_service import AudioService
from services.feature_service import FeatureService


class MusicPipeline:
    """Pipeline for downloading music and adding to database."""

    def __init__(
        self,
        collection_name: str | None = None,
        output_dir: Path | str | None = None,
        cleanup: bool = False,
    ):
        """Initialize pipeline.

        Args:
            collection_name: ChromaDB collection name
            output_dir: Download directory
            cleanup: Whether to delete files after processing
        """
        self.db = ChromaService(collection_name=collection_name)
        self.embedder = MERTService()
        self.downloader = YouTubeService(output_dir=output_dir)
        self.audio_svc = AudioService()
        self.feature_svc = FeatureService()
        self.cleanup = cleanup

        logger.info("MusicPipeline initialized")

    async def process_video(self, url: str) -> dict:
        """Process a single YouTube video.

        Args:
            url: YouTube video URL

        Returns:
            Result dict with status and info
        """
        logger.info(f"Processing video: {url}")

        try:
            # Download
            download_info = await self.downloader.download_async(url)
            video_id = download_info["video_id"]

            # Check for duplicates
            if self.db.song_exists(video_id):
                logger.warning(f"Already exists: {video_id}")
                if self.cleanup:
                    self._cleanup_file(download_info["file_path"])
                return {
                    "status": "skipped",
                    "reason": "duplicate",
                    "video_id": video_id,
                }

            # Validate
            file_path = download_info["file_path"]
            if not file_path or not file_path.exists():
                return {
                    "status": "error",
                    "reason": "file_not_found",
                    "video_id": video_id,
                }

            # Validate audio
            validation = self.audio_svc.validate_audio_file(file_path)
            if not validation["valid"]:
                logger.error(f"Validation failed: {validation['errors']}")
                if self.cleanup:
                    self._cleanup_file(file_path)
                return {
                    "status": "error",
                    "reason": "validation_failed",
                    "errors": validation["errors"],
                }

            # Generate embedding
            embedding = self.embedder.embed_file(file_path)

            # Extract audio features for reranking
            y = self.audio_svc.load_audio(file_path)
            features = self.feature_svc.extract_features(y, self.audio_svc.sample_rate)

            # Prepare metadata
            metadata = {
                "filename": file_path.name,
                "title": download_info["title"],
                "uploader": download_info.get("uploader"),
                "duration_seconds": download_info.get("duration"),
                "upload_date": download_info.get("upload_date"),
                "view_count": download_info.get("view_count"),
                "filepath": str(file_path) if not self.cleanup else None,
            }
            metadata.update(features)

            # Add to database
            self.db.add_song(video_id, embedding.tolist(), metadata)
            logger.success(f"Added: {download_info['title']}")

            # Cleanup if requested
            if self.cleanup:
                self._cleanup_file(file_path)

            return {
                "status": "success",
                "video_id": video_id,
                "title": download_info["title"],
            }

        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            return {"status": "error", "reason": str(e), "url": url}

    async def process_playlist(self, url: str) -> list[dict]:
        """Process all videos from a playlist.

        Args:
            url: YouTube playlist URL

        Returns:
            List of result dicts
        """
        logger.info(f"Processing playlist: {url}")

        try:
            # Get playlist entries
            results = await self.downloader.download_playlist_async(url)

            # Process each result
            processed = []
            for result in results:
                if result["status"] != "success":
                    processed.append(result)
                    continue

                video_id = result["video_id"]
                file_path = result.get("file_path")

                # Check for duplicates
                if self.db.song_exists(video_id):
                    logger.warning(f"Already exists: {video_id}")
                    if self.cleanup and file_path:
                        self._cleanup_file(Path(file_path))
                    processed.append({"status": "skipped", "video_id": video_id})
                    continue

                if not file_path:
                    logger.error(f"Missing file path for {video_id}")
                    processed.append(
                        {
                            "status": "error",
                            "video_id": video_id,
                            "error": "No file path",
                        }
                    )
                    continue

                try:
                    # Validate and embed
                    embedding = self.embedder.embed_file(Path(file_path))

                    # Extract audio features for reranking
                    y = self.audio_svc.load_audio(Path(file_path))
                    features = self.feature_svc.extract_features(
                        y, self.audio_svc.sample_rate
                    )

                    metadata = {
                        "filename": file_path.name,
                        "title": result["title"],
                        "uploader": result.get("uploader"),
                        "duration_seconds": result.get("duration"),
                        "upload_date": result.get("upload_date"),
                        "view_count": result.get("view_count"),
                        "filepath": str(file_path) if not self.cleanup else None,
                    }
                    metadata.update(features)

                    self.db.add_song(video_id, embedding.tolist(), metadata)
                    logger.success(f"Added: {result['title']}")

                    if self.cleanup:
                        self._cleanup_file(file_path)

                    processed.append({"status": "success", "video_id": video_id})

                except Exception as e:
                    logger.error(f"Failed to process {video_id}: {e}")
                    processed.append(
                        {"status": "error", "video_id": video_id, "error": str(e)}
                    )

            return processed

        except Exception as e:
            logger.error(f"Failed to process playlist: {e}")
            return [{"status": "error", "reason": str(e)}]

    def _cleanup_file(self, file_path: Path | None) -> None:
        """Remove a file."""
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                logger.debug(f"Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "total_songs": self.db.count(),
            "collection_name": self.db.collection_name,
        }


async def main():
    """Example usage."""
    pipeline = MusicPipeline()

    # Example: Process a single video
    # result = await pipeline.process_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    # print(result)

    # Example: Process playlist
    playlist_url = (
        "https://www.youtube.com/playlist?list=PLos7xCCYivJ9Oq4pNK9-D1ESAwPkVeveh"
    )
    results = await pipeline.process_playlist(playlist_url)

    # Print summary
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = len(results) - success - skipped

    print("\n📊 Summary:")
    print(f"   Total: {len(results)}")
    print(f"   ✅ Success: {success}")
    print(f"   ⏭️ Skipped: {skipped}")
    print(f"   ❌ Failed: {failed}")

    stats = pipeline.get_stats()
    print(f"\n📀 Database: {stats['total_songs']} songs")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    # Run
    asyncio.run(main())
