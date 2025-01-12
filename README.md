# Music Similarity Search with PANNS and Qdrant

This project implements a music similarity search system using PANNS (Pediatric Automatic Notification and Notification System) for audio embeddings and Qdrant for vector similarity search.

The system allows users to download music from YouTube playlists, generate audio embeddings using a pre-trained PANNS model, store these embeddings in a Qdrant database, and perform similarity searches based on input audio files.

## Repository Structure

```
.
├── add_music_embeddings.py
├── models.py
├── music_download.py
├── query_database.py
├── README.md
├── requirements.yaml
└── utils.py
```

- `add_music_embeddings.py`: Processes audio files and adds embeddings to the Qdrant database.
- `models.py`: Defines PyTorch neural network models for sequence-to-sequence tasks and autoencoding.
- `music_download.py`: Downloads audio files from YouTube playlists and converts them to MP3 format.
- `query_database.py`: Performs similarity searches on the Qdrant database using input audio files.
- `requirements.yaml`: Conda environment configuration file specifying project dependencies.
- `utils.py`: Contains utility functions for audio preprocessing and Qdrant collection creation.

## Usage Instructions

### Installation

1. Ensure you have Conda installed on your system.
2. Create a new Conda environment using the provided `requirements.yaml` file:

```bash
conda env create -f requirements.yaml
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
   QDB_API_TOKEN=your_qdrant_api_token
   ```

   Replace `your_qdrant_endpoint` and `your_qdrant_api_token` with your actual Qdrant database credentials.

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

1. Music Download: YouTube playlist videos are downloaded and converted to MP3 format.
2. Audio Preprocessing: MP3 files are read and preprocessed using librosa.
3. Embedding Generation: The PANNS model generates audio embeddings for each preprocessed audio file.
4. Database Storage: Embeddings are stored in the Qdrant vector database along with song metadata.
5. Similarity Search: User-provided audio files are processed and used to query the database for similar songs.

```
[YouTube Playlist] -> [MP3 Files] -> [Preprocessed Audio] -> [PANNS Model] -> [Audio Embeddings] -> [Qdrant Database]
                                                                                                          ^
                                                                                                          |
[User Input Audio] -> [Preprocessed Audio] -> [PANNS Model] -> [Query Embedding] -------------------------|
```

## Troubleshooting

### Common Issues

1. CUDA Out of Memory Error:
   - Problem: You may encounter a CUDA out of memory error when processing large audio files.
   - Solution: Reduce the batch size or use shorter audio segments. You can modify the `read_preprocess_music` function in `utils.py` to limit the audio length.

2. Qdrant Connection Issues:
   - Problem: Unable to connect to the Qdrant database.
   - Solution: Double-check your `.env` file to ensure the `QDB_ENDPOINT` and `QDB_API_TOKEN` are correct. Verify your network connection and firewall settings.

3. YouTube Download Failures:
   - Problem: `music_download.py` fails to download some videos.
   - Solution: Check if the videos are available in your region. You may need to update the `pytube` library or use a VPN if certain videos are geo-restricted.

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
client = QdrantClient(os.getenv("QDB_ENDPOINT"), api_key=os.getenv("QDB_API_TOKEN"), prefer_grpc=True, timeout=10, debug=True)
```

### Performance Optimization

To optimize performance:

1. Use batched processing when adding embeddings to Qdrant. Modify `add_music_embeddings.py` to process multiple songs in batches.

2. For large datasets, consider using Qdrant's bulk insert functionality instead of individual upserts.

3. Monitor Qdrant's performance using its built-in metrics. You can access these through the Qdrant web interface or API.

4. If query performance is slow, consider adjusting the `ef_search` parameter in Qdrant's search configuration to balance between search speed and recall.