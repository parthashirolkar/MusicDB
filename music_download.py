import os
import re
import asyncio
from yt_dlp import YoutubeDL
# from pydub import AudioSegment

def sanitize_title(title: str) -> str:
    """Sanitize video title to create a valid filename."""
    return re.sub(r'[\\/*?:"<>|]', "", title)

async def download_playlist(playlist_url: str, output_folder: str):
    """Download all videos from a YouTube playlist as MP3 files."""
    # Use yt-dlp to fetch video URLs from the playlist
    ydl_opts = {'extract_flat': True, 'quiet': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        video_urls = [entry['url'] for entry in info['entries']]

    # Process each video asynchronously
    tasks = [
        asyncio.create_task(download_video(video_url, output_folder))
        for video_url in video_urls
    ]
    await asyncio.gather(*tasks)

async def download_video(video_url: str, output_folder: str):
    """Download a single video as an MP3 file."""
    sanitized_title = ""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
            'quiet': True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            sanitized_title = sanitize_title(info['title'])
        
        print(f"Downloaded: {sanitized_title}")
    except Exception as e:
        print(f"Error downloading {video_url}: {e}")    

# Example usage
if __name__ == "__main__":
    playlist_url = "https://www.youtube.com/playlist?list=PLzKILxYC79RDspOFfHUselpBLOVEOJO_T"
    output_folder = "music_files"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    asyncio.run(download_playlist(playlist_url, output_folder))
