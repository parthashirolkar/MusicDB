# Music Similarity Search with wav2vec2 and Qdrant

This project implements a music similarity search system using the wav2vec2 model (facebook/wav2vec2-base) for audio embeddings and Qdrant for vector similarity search.

The system allows users to download music from YouTube playlists (using yt_dlp), generate audio embeddings using a pre-trained wav2vec2 model, store these embeddings in a Qdrant database, and perform similarity searches based on input audio files.

## Repository Structure

```
.
├── add_music_embeddings.py
├── emb_test.ipynb
├── models.py
├── music_download.py
├── pyproject.toml
├── query_database.py
├── README.md
├── requirements.yaml
└── utils.py
```

- `add_music_embeddings.py`: Processes audio files and adds embeddings to the Qdrant database.
- `models.py`: Defines PyTorch neural network models for sequence-to-sequence tasks and autoencoding.
- `music_download.py`: Downloads audio files from YouTube playlists and converts them to MP3 format using yt_dlp.
- `query_database.py`: Performs similarity searches on the Qdrant database using input audio files.
- `requirements.yaml`: Conda environment configuration file specifying project dependencies.
- `pyproject.toml`: Alternative pip/Poetry-based dependency file.
- `utils.py`: Contains utility functions for audio preprocessing and Qdrant collection creation.
- `emb_test.ipynb`: (Optional) Notebook for embedding experiments.

## Usage Instructions

### Installation

1. Ensure you have Conda installed on your system.
2. Create a new Conda environment using the provided `requirements.yaml` file:

```bash
conda env create -f requirements.yaml
```

   Or, to use pip/Poetry (Python >=3.12):

```bash
pip install -r requirements.txt  # or use pyproject.toml with Poetry
```

3. Activate the newly created environment:

```bash
conda activate qdrant-db
```

### Getting Started

1. Set up environment variables:
   Create a `.env` file in the project root directory with the following content:

   ```
   QDB_ENDPOINT=your_qdrant_endpoint
   QDB_API_KEY=your_qdrant_api_key
   ```

   Replace `your_qdrant_endpoint` and `your_qdrant_api_key` with your actual Qdrant database credentials.

2. Download music:
   Edit the `music_download.py` file to specify the desired YouTube playlist URL and output folder. Then run:

   ```bash
   python music_download.py
   ```

3. Add music embeddings to the database:
   Ensure that the downloaded music files are in the `music_files/` directory, then run:

   ```bash
   python add_music_embeddings.py
   ```

4. Perform similarity search:
   Place the audio file you want to use for the search in the `inference_music_files/` directory, then run:

   ```bash
   python query_database.py
   ```

### Configuration Options

- In `add_music_embeddings.py`, you can modify the Qdrant collection name and embedding dimension by changing the `create_quadrant_collection` function call.
- In `query_database.py`, you can adjust the number of similar songs returned by modifying the `limit` parameter in the `client.search` function call.

## Data Flow

The data flow in this project follows these steps:

1. Music Download: YouTube playlist videos are downloaded and converted to MP3 format using yt_dlp.
2. Audio Preprocessing: MP3 files are read and preprocessed using librosa.
3. Embedding Generation: The wav2vec2 model generates audio embeddings for each preprocessed audio file.
4. Database Storage: Embeddings are stored in the Qdrant vector database along with song metadata.
5. Similarity Search: User-provided audio files are processed and used to query the database for similar songs.

```
[YouTube Playlist] -> [MP3 Files] -> [Preprocessed Audio] -> [wav2vec2 Model] -> [Audio Embeddings] -> [Qdrant Database]
                                                                                                          ^
                                                                                                          |
[User Input Audio] -> [Preprocessed Audio] -> [wav2vec2 Model] -> [Query Embedding] ----------------------|
```

## Troubleshooting

### Common Issues

1. CUDA Out of Memory Error:
   - Problem: You may encounter a CUDA out of memory error when processing large audio files.
   - Solution: Reduce the audio length in the `read_preprocess_music` function in `utils.py`.

2. Qdrant Connection Issues:
   - Problem: Unable to connect to the Qdrant database.
   - Solution: Double-check your `.env` file to ensure the `QDB_ENDPOINT` and `QDB_API_KEY` are correct. Verify your network connection and firewall settings.

3. YouTube Download Failures:
   - Problem: `music_download.py` fails to download some videos.
   - Solution: Check if the videos are available in your region. You may need to update the `yt_dlp` library or use a VPN if certain videos are geo-restricted.

### Debugging

To enable verbose logging for better debugging:

1. Add the following lines at the beginning of each Python script:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. For Qdrant-specific debugging, you can enable debug mode when creating the client:

```python
from qdrant_client import QdrantClient
client = QdrantClient(os.getenv("QDB_ENDPOINT"), api_key=os.getenv("QDB_API_KEY"), prefer_grpc=True, timeout=10, debug=True)
```

### Performance Optimization

To optimize performance:

1. For large datasets, consider using Qdrant's bulk insert functionality instead of individual upserts.
2. Monitor Qdrant's performance using its built-in metrics. You can access these through the Qdrant web interface or API.
3. If query performance is slow, consider adjusting the `ef_search` parameter in Qdrant's search configuration to balance between search speed and recall.