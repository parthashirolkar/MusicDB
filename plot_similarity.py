import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def generate_spectrogram_comparison(
    query_path: Path | str,
    match_path: Path | str,
    query_title: str,
    match_title: str,
    similarity: float,
    start_time: float = 30.0,
    duration: float = 30.0,
    sr: int = 24000,
    n_mels: int = 128,
    fmax: int = 8000,
) -> plt.Figure:
    """Generate a side-by-side mel spectrogram comparison plot.

    Args:
        query_path: Path to the query audio file.
        match_path: Path to the matched audio file.
        query_title: Display title for the query.
        match_title: Display title for the match.
        similarity: Similarity score (0-1) to display on the match title.
        start_time: Start time in seconds for the audio snippet.
        duration: Duration in seconds for the audio snippet.
        sr: Sample rate for librosa.
        n_mels: Number of mel bands.
        fmax: Maximum frequency for mel spectrogram.

    Returns:
        matplotlib Figure object.
    """
    query_path = Path(query_path)
    match_path = Path(match_path)

    y_query, _ = librosa.load(query_path, offset=start_time, duration=duration, sr=sr)
    y_match, _ = librosa.load(match_path, offset=start_time, duration=duration, sr=sr)

    S_query = librosa.feature.melspectrogram(y=y_query, sr=sr, n_mels=n_mels, fmax=fmax)
    S_query_db = librosa.power_to_db(S_query, ref=np.max)

    S_match = librosa.feature.melspectrogram(y=y_match, sr=sr, n_mels=n_mels, fmax=fmax)
    S_match_db = librosa.power_to_db(S_match, ref=np.max)

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(14, 8))

    img1 = librosa.display.specshow(
        S_query_db, x_axis="time", y_axis="mel", sr=sr, fmax=fmax, ax=ax[0]
    )
    ax[0].set(title=f"Query: {query_title} ({start_time}s-{start_time + duration}s)")
    ax[0].label_outer()

    librosa.display.specshow(
        S_match_db, x_axis="time", y_axis="mel", sr=sr, fmax=fmax, ax=ax[1]
    )
    ax[1].set(
        title=f"Match: {match_title} ({start_time}s-{start_time + duration}s) | Sim: {similarity:.3f}"
    )

    fig.colorbar(img1, ax=ax, format="%+2.0f dB")
    return fig


def plot_spectrograms():
    from query_database import search_similar_songs

    query_file = "music_files/DJ Snake - Let Me Love You ft. Justin Bieber.mp3"
    print(f"Searching for matches to: {query_file}")

    results = search_similar_songs(query_file, n_results=2)

    if len(results) < 2:
        print("Not enough matches found.")
        return

    match_file = results[1]["metadata"]["filepath"]
    match_title = results[1]["metadata"].get(
        "title", results[1]["metadata"].get("filename")
    )
    similarity = results[1].get("reranked_score", 1 - results[1]["distance"])

    print(f"2nd most similar song: {match_title} (Similarity: {similarity:.3f})")

    fig = generate_spectrogram_comparison(
        query_path=query_file,
        match_path=match_file,
        query_title="Shawn Mendes - In My Blood",
        match_title=match_title,
        similarity=similarity,
    )

    output_img = "similarity_comparison.png"
    fig.savefig(output_img, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_img}")


if __name__ == "__main__":
    plot_spectrograms()
