import os
import uuid
from glob import glob
import numpy as np
import warnings
from torch.utils.data import Dataset, DataLoader
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from qdrant_client import QdrantClient
import librosa  # Add this import at the top
import numpy as np  # Add this import for numpy operations

warnings.simplefilter(action="ignore", category=FutureWarning)
from utils import feature_extractor, create_quadrant_collection, model, read_preprocess_music
from dotenv import load_dotenv
from qdrant_client.http.models import PointStruct


device = "cuda" if torch.cuda.is_available() else "cpu"



def main():
    load_dotenv()

    client = create_quadrant_collection(
        collection_name="song_vector_collection", embedding_size=768
    )

    files = glob("music_files/*.mp3")

    for file in tqdm(files, desc="Processing files"):
        torch.cuda.empty_cache()
        inputs = read_preprocess_music(file)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)
            embedding = embeddings.cpu().numpy().squeeze()

        song_name = os.path.basename(file).split(".")[0]
        payload = {"song_name": song_name}

        client.upsert(
            collection_name="song_vector_collection",
            points=[
                PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=payload),
            ],
        )




if __name__ == "__main__":
    main()
