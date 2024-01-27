import os
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
    
    
    print(f"{youtube.title} downloaded and converted to MP3.")

if __name__ == "__main__":
    
    playlist_url = "https://youtube.com/playlist?list=PLMC9KNkIncKtPzgY-5rmhvj7fax8fdxoj&si=Zw7S450__EF3oiRB"
    output_folder = "music_files"

    download_playlist(playlist_url, output_folder)
    # Remove all the mp4 files after they have been converted to mp3
    os.system("rm -rf music_files/*.mp4")
