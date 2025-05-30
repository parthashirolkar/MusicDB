import os
import warnings
import torch

warnings.simplefilter(action="ignore", category=FutureWarning)
from utils import read_preprocess_music, model
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(os.getenv("QDB_ENDPOINT"), api_key=os.getenv("QDB_API_TOKEN"))

selected_file = os.listdir("inference_music_files")[-1]
inputs = read_preprocess_music(os.path.join("inference_music_files", selected_file))

print("User input song: ", selected_file)

with torch.no_grad():
    embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    user_embedding = embeddings.cpu().numpy().squeeze()

search_result = client.search(
    collection_name="song_vector_collection", query_vector=user_embedding, limit=5
)

for result in search_result:
    print(
        f"ID: {result.id}, Song: {result.payload['song_name']}, Score: {result.score}"
    )

client.close()
