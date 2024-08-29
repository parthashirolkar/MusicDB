import os
from music_download import download_video

import numpy as np
import librosa
import faiss

def extract_features(file_path: str, scaler=None) -> np.ndarray:
    y, sr = librosa.load(file_path, duration=60)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    features = np.vstack([mfccs, chroma, spectral_contrast])

    return np.mean(features, axis=1)

def create_index(feature_dim: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(feature_dim)

    return index

def add_vectors(index: faiss.IndexFlatL2, vectors: np.ndarray) -> None:
    if len(vectors.shape) == 1:
        vectors = vectors.reshape(1, -1)

    vectors = vectors.astype(np.float32)
    index.add(vectors)


def search_similar_music(index: faiss.IndexFlatL2, query_vector: np.ndarray, k: int=2):
    query_vector = query_vector.reshape(1,-1).astype(np.float32)
    distances, indices = index.search(query_vector, k=k)
    return distances, indices




music_db = {}

for i, song in enumerate(os.listdir("music_files/"), start=1):
    if song.endswith(".mp3"):
        music_db.update({i:song})


download_video(video_url="https://www.youtube.com/watch?v=qFLhGq0060w&ab_channel=TheWeekndVEVO", output_folder="inference_music_files")
user_input_features = extract_features(os.path.join("inference_music_files",os.listdir("inference_music_files")[0]))

feature_dim = len(user_input_features)
index = create_index(feature_dim)

for song, file_path in music_db.items():
    song_features = extract_features(os.path.join("music_files",file_path))
    add_vectors(index, song_features)

k = 2  # Top 2 recommendations
distances, indices = search_similar_music(index, user_input_features, k=k)
top_recommendations = [list(music_db.keys())[i] for i in indices[0]]

print("Top 2 recommended songs:")
for i, idx in enumerate(top_recommendations, 1):
    song_name = music_db[idx]
    similarity_score = distances[0][i-1]  # Since i starts from 1
    print(f"{i}. Song: {song_name}, Similarity Score: {similarity_score:.4f}")
