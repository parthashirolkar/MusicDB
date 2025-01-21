import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
from utils import read_preprocess_music
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from panns_inference import AudioTagging

load_dotenv()


client = QdrantClient(os.getenv("QDB_ENDPOINT"), api_key=os.getenv("QDB_API_TOKEN"))

at = AudioTagging(checkpoint_path=None, device="cuda")

selected_file = os.listdir("inference_music_files")[-1]
user_input = read_preprocess_music(os.path.join("inference_music_files", selected_file))

print("User input song: ", selected_file)
(_, user_embedding) = at.inference(user_input)
user_embedding = np.squeeze(user_embedding, axis=0)

search_result = client.search(
    collection_name="song_vector_collection", query_vector=user_embedding, limit=5
)

for result in search_result:
    print(
        f"ID: {result.id}, Song: {result.payload['song_name']}, Score: {result.score}"
    )

client.close()
