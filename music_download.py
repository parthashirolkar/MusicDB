import os
import time
import argparse
from pytube import YouTube, Playlist
from pytube.cli import on_progress
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor

def download_playlist(playlist_url, output_folder):
    playlist = Playlist(playlist_url)
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(download_video, playlist.video_urls, [output_folder]*len(playlist.video_urls))

def download_video(video_url, output_folder):
    youtube = YouTube(video_url, on_progress_callback=on_progress)
    video_stream = youtube.streams.filter(only_audio=True).first()
    
    output_file = os.path.join(output_folder, f"{youtube.title}.mp3")
    video_stream.download(output_folder)
    
    audio = AudioSegment.from_file(os.path.join(output_folder, video_stream.default_filename), format="mp4")
    audio.export(output_file, format="mp3")
    [os.remove(os.path.join(output_folder,file)) for file in os.listdir(output_folder) if file.endswith(".mp4")]
    print(f"{youtube.title} downloaded and converted to MP3.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and convert a YouTube playlist to MP3.")
    parser.add_argument("playlist_url", help="URL of the YouTube playlist")
    parser.add_argument("output_folder", help="Output folder for downloaded and converted files")

    args = parser.parse_args()
    start_time = time.perf_counter()
    download_playlist(args.playlist_url, args.output_folder)
    
    # Remove all the mp4 files after they have been converted to mp3
    os.system(f"rm -rf {args.output_folder}/*.mp4")
    print(f"Running time of the script: {round(time.perf_counter() - start_time, 3)}")