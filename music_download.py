import os
from pytube import YouTube, Playlist
from pytube.cli import on_progress
from pydub import AudioSegment

def download_playlist(playlist_url, output_folder):
    playlist = Playlist(playlist_url)
    
    for video_url in playlist.video_urls:
        download_video(video_url, output_folder)

def download_video(video_url, output_folder):
    youtube = YouTube(video_url, on_progress_callback=on_progress)
    video_stream = youtube.streams.filter(only_audio=True).first()
    
    
    
    output_file = f"{output_folder}/{youtube.title}.mp3"
    video_stream.download(output_folder)
    
    
    audio = AudioSegment.from_file(os.path.join(output_folder,f"{youtube.title}.mp4"), format="mp4")
    audio.export(output_file, format="mp3")



if __name__ == "__main__":
    # Example usage
    playlist_url = "https://youtube.com/playlist?list=PLMC9KNkIncKtPzgY-5rmhvj7fax8fdxoj&si=Zw7S450__EF3oiRB"
    output_folder = "music_files"

    download_playlist(playlist_url, output_folder)
