"""Outdated script.... Doesnt Work anymore"""


import os
from music_download import download_video
from tqdm.auto import tqdm
import numpy as np
import librosa
import faiss
from models import BiLSTMSeq2Seq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast

device = "cuda" if torch.cuda.is_available() else "cpu"


model = BiLSTMSeq2Seq(input_dim=50, hidden_dim_1=64, hidden_dim_2=256).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))

index = faiss.IndexFlatL2(512)



def preprocess_music(file_path, n_mfcc=50):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = librosa.util.normalize(mfcc)
    return np.expand_dims(mfcc.T, axis=0)  # Transpose to shape (timesteps, features)

def pad_and_truncate_sequences(sequences, maxlen=5000, padding_value=0.0):
    """
    Pads and truncates sequences to the specified maximum length.

    Args:
        sequences (list of tensors): List of variable-length sequences (each is a 1D tensor).
        maxlen (int): Maximum length to pad or truncate to.
        padding_value (float): Value used for padding (default is 0.0).

    Returns:
        Tensor: Padded and truncated tensor with shape (batch_size, maxlen, feature_dim).
    """
    sequences = torch.tensor(sequences, dtype=torch.float32)
    # Convert list of sequences to a batch tensor using pad_sequence
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)

    # Truncate sequences if they exceed maxlen
    if padded_sequences.size(1) > maxlen:
        padded_sequences = padded_sequences[:, :maxlen, :]
    # Pad sequences to maxlen if they are shorter
    elif padded_sequences.size(1) < maxlen:
        # Create a tensor of shape (batch_size, maxlen, feature_dim) filled with padding_value
        pad_shape = (padded_sequences.size(0), maxlen - padded_sequences.size(1), padded_sequences.size(2))
        padding = torch.full(pad_shape, padding_value)
        padded_sequences = torch.cat([padded_sequences, padding], dim=1)

    return padded_sequences



def create_music_embeddings(mfccs: torch.tensor) -> torch.tensor:
    mfccs = mfccs.to(device)
    model.eval()

    with torch.no_grad(),autocast(dtype=torch.float32):
        embeddings = model.encoder(mfccs)
    embeddings = embeddings.squeeze(0)

    return embeddings.detach().cpu().numpy()



def add_vectors(index: faiss.IndexFlatL2, vectors: np.ndarray) -> None:
    if len(vectors.shape) == 1:
        vectors = vectors.reshape(1, -1)

    vectors = vectors.astype(np.float32)
    index.add(vectors)


def search_similar_music(index: faiss.IndexFlatL2, query_vector: np.ndarray, k: int=2):
    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
    distances, indices = index.search(query_vector, k=k)
    return distances, indices

music_db = {}

for i, song in enumerate(os.listdir("music_files/"), start=1):
    if song.endswith(".mp3"):
        music_db.update({i:song})



for key, file_path in tqdm(music_db.items()):
    mfccs = preprocess_music(os.path.join("music_files", file_path))
    mfccs = pad_and_truncate_sequences(mfccs, maxlen=5000, padding_value=0.0)
    embeddings = create_music_embeddings(mfccs)
    add_vectors(index, embeddings)

# download_video(video_url="https://www.youtube.com/watch?v=qFLhGq0060w&ab_channel=TheWeekndVEVO", output_folder="inference_music_files")
selected_file = os.listdir("inference_music_files")[-1]
user_input = preprocess_music(os.path.join("inference_music_files",selected_file))
print("User input song: ", selected_file)

user_input = pad_and_truncate_sequences(user_input, maxlen=5000, padding_value=0.0)
user_embeddings = create_music_embeddings(user_input)


k = 2  # Top 2 recommendations
distances, indices = search_similar_music(index, user_embeddings, k=k)
print("Distances: ", distances)
print("Indices: ", indices)
top_recommendations = [list(music_db.keys())[i] for i in indices[0]]

print("Top 2 recommended songs:")
for i, idx in enumerate(top_recommendations, 1):
    song_name = music_db[idx]
    similarity_score = distances[0][i-1]  # Since i starts from 1
    print(f"{i}. Song: {song_name}, Similarity Score: {similarity_score:.4f}")
