import os
import warnings
from music_download import download_video
import numpy as np
import librosa
from librosa import feature
import faiss

warnings.simplefilter(action="ignore", category=FutureWarning)

# def extract_features(file_path: str) -> np.ndarray:
#     y, sr = librosa.load(file_path, duration=60)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr)
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#     spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

#     features = np.vstack([mfccs, chroma, spectral_contrast])

#     mean = np.mean(features, axis=1, keepdims=True)
#     std = np.std(features, axis=1, keepdims=True)
#     scaled_features = (features - mean) / std

#     return scaled_features.astype(np.float32)

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=60)
    fn_list_i = [
    feature.chroma_stft,
    feature.spectral_centroid,
    feature.spectral_bandwidth,
    feature.spectral_rolloff
    ]
    
    fn_list_ii = [
    feature.rms,
    feature.zero_crossing_rate
    ]

    feat_vect_i = [np.mean(funct(y=y,sr=sr)) for funct in fn_list_i]
    feat_vect_ii = [np.mean(funct(y=y)) for funct in fn_list_ii] 
    feature_vector = feat_vect_i + feat_vect_ii 
    return np.array(feature_vector).astype(np.float32)

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


download_video(video_url="https://www.youtube.com/watch?v=w9M7ydpcDSw&list=RDGMEMHDXYb1_DDSgDsobPsOFxpAVMw9M7ydpcDSw&start_radio=1&ab_channel=TravisScottVEVO", output_folder="inference_music_files")
user_input_features = extract_features(os.path.join("inference_music_files",os.listdir("inference_music_files")[0]))
user_input_features = user_input_features.reshape(1,-1)
feature_dim = user_input_features.shape[1]
index = create_index(feature_dim)

for song, file_path in music_db.items():
    song_features = extract_features(os.path.join("music_files",file_path))

    add_vectors(index, song_features)


faiss.write_index(index, "indexes/preliminary.index")
k = 5  # Top 5 recommendations
distances, indices = search_similar_music(index, user_input_features.astype(np.float32), k=k)
top_recommendations = [list(music_db.keys())[i] for i in indices[0]]

print("Top 5 recommended songs:")

recommendations_with_scores = list(zip(indices[0], distances[0]))

sorted_recommendations = sorted(recommendations_with_scores, key=lambda x: x[1], reverse=True)

for i, (idx, similarity_score) in enumerate(sorted_recommendations[:k], 1):
    song_name = music_db[idx]
    print(f"{i}. Song: {song_name}, Similarity Score: {np.sqrt(similarity_score):.4f}")
