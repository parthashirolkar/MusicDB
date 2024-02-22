import os
import warnings
from joblib import Parallel, delayed
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

def extract_features_parallel(file_paths):
    return np.array(Parallel(n_jobs=-1)(delayed(extract_features)(file_path) for file_path in file_paths)).astype(np.float32)


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

def create_index_ivf(features: np.ndarray, nlist: int = 5) -> faiss.IndexIVFFlat:
    feature_dim = features.shape[1]
    quantizer = faiss.IndexFlatL2(feature_dim)
    index = faiss.IndexIVFFlat(quantizer, feature_dim, nlist)
    index.train(features.astype(np.float32))  # Train the index with the features
    index.add(features.astype(np.float32))    # Add the features to the index
    return index




def search_similar_music(index: faiss.IndexFlatL2, query_vector: np.ndarray, k: int=2):
    query_vector = query_vector.reshape(1,-1).astype(np.float32)
    distances, indices = index.search(query_vector, k=k)
    return distances, indices




music_db = {}

for i, song in enumerate(os.listdir("music_files/"), start=1):
    if song.endswith(".mp3"):
        music_db.update({i:song})


print("Started Feature Extraction...")
file_paths = [os.path.join("music_files", file_path) for file_path in music_db.values()]
music_features = np.array(extract_features_parallel(file_paths))
print(f"Extracted features of {len(file_paths)} songs.")

print(music_features)

feature_dim = music_features.shape[1]
index = create_index_ivf(music_features)
print("Vector database creation finished, writing to disk...")


# download_video(video_url="https://www.youtube.com/watch?v=B6_iQvaIjXw&ab_channel=ArianaGrandeVevo", output_folder="inference_music_files")
inf_files = [os.path.join("inference_music_files", file_name) for file_name in os.listdir("inference_music_files")]

user_input_features = extract_features(inf_files[1])
user_input_features = user_input_features.reshape(1,-1)
print(user_input_features.shape)
print(index.d)
# faiss.write_index(index, "indexes/preliminary.index")
k = 5  # Top 5 recommendations
distances, indices = search_similar_music(index, user_input_features.astype(np.float32), k=k)
print("Searching for similar music...")
top_recommendations = [list(music_db.keys())[i] for i in indices[0]]

print("Top 5 recommended songs:")

recommendations_with_scores = list(zip(indices[0], distances[0]))
sorted_recommendations = sorted(recommendations_with_scores, key=lambda x: x[1], reverse=True)

for i, (idx, similarity_score) in enumerate(sorted_recommendations[:k], 1):
    song_name = music_db[idx]
    print(f"{i}. Song: {song_name}, Similarity Score: {np.sqrt(similarity_score):.4f}")
