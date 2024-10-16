import os
from utils import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from dotenv import load_dotenv
from models import BiLSTMSeq2Seq
import torch
from qdrant_client import QdrantClient

load_dotenv()


client = QdrantClient(os.getenv("QDB_ENDPOINT"), api_key=os.getenv("QDB_API_TOKEN"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BiLSTMSeq2Seq(input_dim=50, hidden_dim_1=64, hidden_dim_2=256).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))

selected_file = os.listdir("inference_music_files")[-1]
user_input = preprocess_music(os.path.join("inference_music_files",selected_file))
print("User input song: ", selected_file)

user_input = pad_and_truncate_sequences(user_input, maxlen=5000, padding_value=0.0)
user_embeddings = create_music_embeddings(user_input, model)

search_result = client.search(
    collection_name="test_collection",
    query_vector=user_embeddings.tolist(),
    limit=5
)

for result in search_result:
    print(f"ID: {result.id}, Song: {result.payload['song_name']}, Score: {result.score}")

client.close()