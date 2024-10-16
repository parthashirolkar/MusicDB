import os
from utils import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from dotenv import load_dotenv
from tqdm.auto import tqdm
from models import BiLSTMSeq2Seq
import torch
from torch.nn.utils.rnn import pad_sequence


from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Distance, VectorParams


load_dotenv()

client = QdrantClient(os.getenv("QDB_ENDPOINT"), api_key=os.getenv("QDB_API_TOKEN"))
if not client.collection_exists("test_collection"):
    client.create_collection(
            collection_name="test_collection",
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BiLSTMSeq2Seq(input_dim=50, hidden_dim_1=64, hidden_dim_2=256).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))


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
