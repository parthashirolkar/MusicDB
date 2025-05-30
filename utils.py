import os
from dotenv import load_dotenv
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModel

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize wav2vec2 model and feature extractor
model = AutoModel.from_pretrained("facebook/wav2vec2-base").to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")


def read_preprocess_music(file_path: str):
    audio, sr = librosa.load(file_path)
    # Resample to 16kHz as required by wav2vec2
    resampled_audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16_000)

    inputs = feature_extractor(
        resampled_audio,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
        truncation=True,
        max_length=16_000,
    ).to(device)

    return inputs


def create_quadrant_collection(collection_name: str, embedding_size: int):
    client = QdrantClient(os.getenv("QDB_ENDPOINT"), api_key=os.getenv("QDB_API_KEY"))

    if client.collection_exists(collection_name):
        return client  # Collection already exists, return the client object

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
    )
    return client
