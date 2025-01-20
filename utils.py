import os
from dotenv import load_dotenv
import numpy as np
import librosa
import torch

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"


def read_preprocess_music(file_path: str) -> np.ndarray:
    (audio, _) = librosa.load(file_path)
    audio = audio[None, :]

    return audio

def create_quadrant_collection(collection_name: str, embedding_size: int):
    client = QdrantClient(os.getenv("QDB_ENDPOINT"), api_key=os.getenv("QDB_API_TOKEN"))

    if client.collection_exists(collection_name):
        return client  # Collection already exists, return the client object

    client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
    )
    return client  # Return the client object after creating the collection
