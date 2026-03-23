import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from query_database import search_similar_songs


def plot_spectrograms():
    query_file = "music_files/DJ Snake - Let Me Love You ft. Justin Bieber.mp3"
    print(f"Searching for matches to: {query_file}")

    # Get top 2 results. Index 0 is the song itself, index 1 is the 2nd most similar.
    results = search_similar_songs(query_file, n_results=2)

    if len(results) < 2:
        print("Not enough matches found.")
        return

    match_file = results[1]["metadata"]["filepath"]
    match_title = results[1]["metadata"].get(
        "title", results[1]["metadata"].get("filename")
    )
    similarity = 1 - results[1]["distance"]

    print(f"2nd most similar song: {match_title} (Similarity: {similarity:.3f})")

    # Load 30 seconds of audio from the middle of the songs (e.g., from 30s to 60s)
    # This avoids long silent intros and gets right to the music
    duration_to_plot = 30
    start_time = 30

    print("Loading audio files (taking a 30s snippet)...")
    y_query, sr = librosa.load(
        query_file, offset=start_time, duration=duration_to_plot, sr=24000
    )
    y_match, _ = librosa.load(
        match_file, offset=start_time, duration=duration_to_plot, sr=24000
    )

    print("Computing Mel spectrograms...")
    S_query = librosa.feature.melspectrogram(y=y_query, sr=sr, n_mels=128, fmax=8000)
    S_query_db = librosa.power_to_db(S_query, ref=np.max)

    S_match = librosa.feature.melspectrogram(y=y_match, sr=sr, n_mels=128, fmax=8000)
    S_match_db = librosa.power_to_db(S_match, ref=np.max)

    print("Generating plot...")
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(14, 8))

    img1 = librosa.display.specshow(
        S_query_db, x_axis="time", y_axis="mel", sr=sr, fmax=8000, ax=ax[0]
    )
    ax[0].set(title=f"Query: Shawn Mendes - In My Blood (30s-60s)")
    ax[0].label_outer()

    img2 = librosa.display.specshow(
        S_match_db, x_axis="time", y_axis="mel", sr=sr, fmax=8000, ax=ax[1]
    )
    ax[1].set(title=f"Match: {match_title} (30s-60s) | Sim: {similarity:.3f}")

    fig.colorbar(img1, ax=ax, format="%+2.0f dB")

    output_img = "similarity_comparison.png"
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_img}")


if __name__ == "__main__":
    plot_spectrograms()
