import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import read_preprocess_music, create_quadrant_collection
import numpy as np
from dotenv import load_dotenv
from panns_inference import AudioTagging
import torch

from qdrant_client.http.models import PointStruct


load_dotenv()

client = create_quadrant_collection(collection_name="panns_collection", dim=2048)

at = AudioTagging(checkpoint_path=None, device='cuda')


# PANNS Inference
for i, song in enumerate(os.listdir("music_files/"), start=1):
    torch.cuda.empty_cache()
    print(f"Processing song {i}: {song}")
    file_path = os.path.join("music_files/", song)
    audio = read_preprocess_music(file_path)

    (_, embedding) = at.inference(audio)
    embedding = np.squeeze(embedding, axis=0)

    operation_info = client.upsert(
        collection_name="panns_collection",
        wait=True,
        points=[
            PointStruct(id=i, vector=embedding.tolist(), payload={"song_name": song}),
        ]
    )

client.close()
