import os
from utils import read_preprocess_music
import numpy as np
from dotenv import load_dotenv
from panns_inference import AudioTagging
import torch

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Distance, VectorParams


load_dotenv()

client = QdrantClient(os.getenv("QDB_ENDPOINT"), api_key=os.getenv("QDB_API_TOKEN"))
if not client.collection_exists("panns_collection"):
    client.create_collection(
            collection_name="panns_collection",
            vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
    )

at = AudioTagging(checkpoint_path=None, device='cuda')


# for i, song in enumerate(os.listdir("music_files/"), start=1):
#     print(f"Processing song {i}: {song}")
#     file_path = os.path.join("music_files/", song)
#     mfccs = preprocess_music(file_path)
#     padded_mfccs = pad_and_truncate_sequences(mfccs)
#     embeddings = create_music_embeddings(padded_mfccs)
#     operation_info = client.upsert(
#         collection_name="panns_collection",
#         wait=True,
#         points=[
#             PointStruct(id=i, vector=embeddings.tolist(), payload={"song_name": song}),
#         ]
#     )

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

print(operation_info)   
client.close()
