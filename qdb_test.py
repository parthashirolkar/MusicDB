import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# from music_download import download_video
from tqdm.auto import tqdm
import numpy as np
import librosa
from models import BiLSTMSeq2Seq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast


from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Distance, VectorParams


client = QdrantClient(":memory:")
if not client.collection_exists("test_collection"):
    client.create_collection(
            collection_name="test_collection",
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BiLSTMSeq2Seq(input_dim=50, hidden_dim_1=64, hidden_dim_2=256).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))


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
    embeddings = torch.mean(embeddings, dim=0)

    return embeddings.detach().cpu().numpy()



for i, song in enumerate(os.listdir("music_files/"), start=1):
    print(f"Processing song {i}: {song}")
    file_path = os.path.join("music_files/", song)
    mfccs = preprocess_music(file_path)
    padded_mfccs = pad_and_truncate_sequences(mfccs)
    embeddings = create_music_embeddings(padded_mfccs)
    operation_info = client.upsert(
        collection_name="test_collection",
        wait=True,
        points=[
            PointStruct(id=i, vector=embeddings.tolist(), payload={"song_name": song}),
        ]
    )
print(operation_info)   

selected_file = os.listdir("inference_music_files")[1]
user_input = preprocess_music(os.path.join("inference_music_files",selected_file))
print("User input song: ", selected_file)

user_input = pad_and_truncate_sequences(user_input, maxlen=5000, padding_value=0.0)
user_embeddings = create_music_embeddings(user_input)

search_result = client.search(
    collection_name="test_collection",
    query_vector=user_embeddings.tolist(),
    limit=5
)

for result in search_result:
    print(f"ID: {result.id}, Song: {result.payload['song_name']}, Score: {result.score}")
