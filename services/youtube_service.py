"""YouTube download service."""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable
from loguru import logger
from yt_dlp import YoutubeDL

from core.config import get_settings
from core.exceptions import DownloadError


class YouTubeService:
    """Service for downloading YouTube videos."""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        max_workers: int = 3,
    ):
        """Initialize YouTube service.

        Args:
            output_dir: Directory to save downloads
            max_workers: Max concurrent downloads (default 3 to avoid rate limits)
        """
        settings = get_settings()
        self.output_dir = (
            Path(output_dir) if output_dir else settings.download.output_dir
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._semaphore = asyncio.Semaphore(max_workers)

    def _get_ydl_opts(self, progress_hook: Callable | None = None) -> dict:
        """Get yt-dlp options."""
        settings = get_settings()

        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(self.output_dir / "%(title)s.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": settings.download.audio_format,
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "no_warnings": True,
        }

        if progress_hook:
            opts["progress_hooks"] = [progress_hook]

        return opts

    @staticmethod
    def extract_video_id(url: str) -> str | None:
        """Extract YouTube video ID from URL or bare video ID.

        Args:
            url: YouTube URL or bare video ID

        Returns:
            Video ID or None
        """
        patterns = [
            r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\s?]+)",
            r"youtube\.com/shorts/([^&\s?]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        if re.match(r"^[a-zA-Z0-9_-]{11}$", url):
            return url

        return None

    def get_info(self, url: str) -> dict:
        """Get video info without downloading.

        Args:
            url: YouTube video URL

        Returns:
            Dict with video_id, title, and other info

        Raises:
            DownloadError: If info extraction fails
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            raise DownloadError(f"Invalid YouTube URL: {url}")

        try:
            opts = {"quiet": True, "no_warnings": True, "extract_flat": False}
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info is None:
                    raise DownloadError("Failed to extract video info")
                return {
                    "video_id": video_id,
                    "title": info.get("title", "unknown"),
                    "duration": info.get("duration"),
                    "uploader": info.get("uploader"),
                }
        except Exception as e:
            raise DownloadError(f"Failed to get video info: {e}")

    def find_cached_file(self, video_id: str, title: str) -> Path | None:
        """Find a cached file by title first, then video ID.

        Args:
            video_id: YouTube video ID
            title: Video title

        Returns:
            Path to cached file or None
        """
        settings = get_settings()
        ext = settings.download.audio_format

        title_clean = re.sub(r'[\\/*?:"<>|]', "", title)
        title_file = self.output_dir / f"{title_clean}.{ext}"
        if title_file.exists():
            return title_file

        for file in self.output_dir.glob(f"*.{ext}"):
            if title_clean.lower() in file.stem.lower():
                return file

        video_id_file = self.output_dir / f"{video_id}.{ext}"
        if video_id_file.exists():
            return video_id_file

        for file in self.output_dir.glob(f"*.{ext}"):
            if video_id in file.stem:
                return file

        return None

    def download(self, url: str, progress_hook: Callable | None = None) -> dict:
        """Download a single video.

        Args:
            url: YouTube video URL
            progress_hook: Optional progress callback

        Returns:
            Download info dict

        Raises:
            DownloadError: If download fails
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            raise DownloadError(f"Invalid YouTube URL: {url}")

        try:
            logger.info(f"Downloading: {url}")

            opts = self._get_ydl_opts(progress_hook)

            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)

                if info is None:
                    raise DownloadError("Failed to extract video info")

                title = info.get("title", "unknown")

                # Get the actual file path from yt-dlp info
                file_path = None
                if "requested_downloads" in info and info["requested_downloads"]:
                    path_str = info["requested_downloads"][0].get("filepath")
                    if path_str:
                        file_path = Path(path_str)

                # Fallback to title-based search if path not found
                if not file_path or not file_path.exists():
                    file_path = self._find_downloaded_file(title)

                if not file_path or not file_path.exists():
                    raise DownloadError(
                        f"Could not locate downloaded file for: {title}"
                    )

                result = {
                    "video_id": video_id,
                    "title": title,
                    "file_path": file_path,
                    "duration": info.get("duration"),
                    "uploader": info.get("uploader"),
                    "upload_date": info.get("upload_date"),
                    "view_count": info.get("view_count"),
                    "description": info.get("description", "")[:500],  # Truncate
                }

                logger.info(f"Downloaded: {title}")
                return result

        except Exception as e:
            raise DownloadError(f"Failed to download {url}: {e}")

    async def download_async(
        self, url: str, progress_hook: Callable | None = None
    ) -> dict:
        """Download a single video asynchronously.

        Args:
            url: YouTube video URL
            progress_hook: Optional progress callback

        Returns:
            Download info dict

        Raises:
            DownloadError: If download fails
        """
        # Use semaphore to limit concurrent downloads
        async with self._semaphore:
            # Run blocking download in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor, self.download, url, progress_hook
            )

    def download_playlist(
        self, url: str, progress_hook: Callable | None = None
    ) -> list[dict]:
        """Download all videos from a playlist (synchronous).

        Args:
            url: YouTube playlist URL
            progress_hook: Optional progress callback

        Returns:
            List of download info dicts
        """
        try:
            logger.info(f"Extracting playlist: {url}")

            # Get playlist entries
            opts = {"extract_flat": True, "quiet": True}

            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                entries = info.get("entries", [])

            if not entries:
                raise DownloadError("No videos found in playlist")

            logger.info(f"Found {len(entries)} videos in playlist")

            # Download each video
            results = []
            for entry in entries:
                video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                try:
                    result = self.download(video_url, progress_hook)
                    results.append({"status": "success", **result})
                except Exception as e:
                    logger.error(f"Failed to download {video_url}: {e}")
                    results.append(
                        {"status": "failed", "video_id": entry["id"], "error": str(e)}
                    )

            return results

        except Exception as e:
            raise DownloadError(f"Failed to process playlist: {e}")

    async def download_playlist_async(
        self,
        url: str,
        progress_hook: Callable | None = None,
        concurrent: bool = True,
    ) -> list[dict]:
        """Download all videos from a playlist asynchronously.

        Args:
            url: YouTube playlist URL
            progress_hook: Optional progress callback
            concurrent: If True, download videos concurrently. If False, sequential.

        Returns:
            List of download info dicts
        """
        try:
            logger.info(f"Extracting playlist: {url}")

            # Get playlist entries (this is fast, so do it synchronously)
            opts = {"extract_flat": True, "quiet": True}

            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                entries = info.get("entries", [])

            if not entries:
                raise DownloadError("No videos found in playlist")

            logger.info(f"Found {len(entries)} videos in playlist")

            if concurrent:
                # Download all videos concurrently with semaphore limiting
                tasks = [
                    self._download_single_async(
                        f"https://www.youtube.com/watch?v={entry['id']}",
                        entry["id"],
                        progress_hook,
                    )
                    for entry in entries
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Convert exceptions to error dicts
                processed = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        processed.append(
                            {
                                "status": "failed",
                                "video_id": entries[i]["id"],
                                "error": str(result),
                            }
                        )
                    else:
                        processed.append(result)
                return processed
            else:
                # Sequential download (safer for rate limiting)
                results = []
                for entry in entries:
                    video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                    try:
                        result = await self.download_async(video_url, progress_hook)
                        results.append({"status": "success", **result})
                    except Exception as e:
                        logger.error(f"Failed to download {video_url}: {e}")
                        results.append(
                            {
                                "status": "failed",
                                "video_id": entry["id"],
                                "error": str(e),
                            }
                        )
                return results

        except Exception as e:
            raise DownloadError(f"Failed to process playlist: {e}")

    async def _download_single_async(
        self, url: str, video_id: str, progress_hook: Callable | None = None
    ) -> dict:
        """Helper to download a single video and wrap result.

        Args:
            url: Video URL
            video_id: Video ID for error reporting
            progress_hook: Optional progress callback

        Returns:
            Result dict with status
        """
        try:
            result = await self.download_async(url, progress_hook)
            return {"status": "success", **result}
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return {"status": "failed", "video_id": video_id, "error": str(e)}

    def _find_downloaded_file(self, title: str) -> Path | None:
        """Find the downloaded file by title.

        Args:
            title: Video title

        Returns:
            Path to file or None
        """
        settings = get_settings()
        ext = settings.download.audio_format

        # Try exact match
        exact = self.output_dir / f"{title}.{ext}"
        if exact.exists():
            return exact

        # Try sanitized match
        sanitized = re.sub(r'[\\/*?:"<>|]', "", title)
        sanitized_path = self.output_dir / f"{sanitized}.{ext}"
        if sanitized_path.exists():
            return sanitized_path

        # Search directory
        for file in self.output_dir.glob(f"*.{ext}"):
            if sanitized.lower() in file.stem.lower():
                return file

        return None
