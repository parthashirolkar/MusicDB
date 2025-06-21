# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a music similarity search system using wav2vec2 embeddings and Qdrant vector database. The system downloads music from YouTube, generates audio embeddings using Facebook's wav2vec2-base model, stores them in Qdrant, and performs similarity searches.

## Environment Setup

The project uses both conda and uv for dependency management:

**Conda environment (recommended):**
```bash
conda env create -f requirements.yaml
conda activate qdrant-db
```

**Alternative with uv/pip:**
```bash
uv sync  # or pip install from pyproject.toml
```

## Required Environment Variables

Create a `.env` file with:
```
QDB_ENDPOINT=your_qdrant_endpoint
QDB_API_KEY=your_qdrant_api_key
```

## Common Development Commands

**Linting:**
```bash
ruff check .
ruff format .
```

**Running the system:**
```bash
# 1. Download music from YouTube playlists
python music_download.py

# 2. Generate embeddings and populate database
python add_music_embeddings.py

# 3. Query for similar songs
python query_database.py
```

**Jupyter notebook for experimentation:**
```bash
jupyter lab emb_test.ipynb
```

## Architecture Overview

### Core Components

- **`utils.py`**: Core utilities for audio preprocessing and Qdrant client setup
  - `read_preprocess_music()`: Loads audio, resamples to 16kHz, creates wav2vec2 inputs
  - `create_quadrant_collection()`: Sets up Qdrant collection with cosine similarity
  - Initializes wav2vec2 model (`facebook/wav2vec2-base`) and feature extractor

- **`add_music_embeddings.py`**: Batch processing pipeline
  - Processes all MP3 files in `music_files/` directory
  - Generates 768-dimensional embeddings using wav2vec2
  - Stores embeddings in Qdrant collection "song_vector_collection"

- **`query_database.py`**: Search functionality
  - Processes query audio from `inference_music_files/` directory
  - Performs cosine similarity search against stored embeddings

- **`models.py`**: PyTorch LSTM encoder-decoder models (appears to be experimental/unused)

- **`music_download.py`**: YouTube playlist downloader using yt_dlp

### Data Flow

```
YouTube Playlist → MP3 Files → Audio Preprocessing → wav2vec2 Model → Embeddings → Qdrant DB
                                                                                      ↑
Query Audio → Audio Preprocessing → wav2vec2 Model → Query Embedding ────────────────┘
```

### Key Technical Details

- **Audio Processing**: Uses librosa for loading/resampling, wav2vec2 requires 16kHz sampling rate
- **Embedding Model**: Facebook wav2vec2-base produces 768-dimensional vectors
- **Vector DB**: Qdrant with cosine similarity distance metric
- **GPU Support**: Automatically detects CUDA availability, falls back to CPU
- **Memory Management**: Includes `torch.cuda.empty_cache()` for large audio processing

### Directory Structure

- `music_files/`: Audio files for embedding generation
- `inference_music_files/`: Query audio files for similarity search
- Audio length is truncated to 16,000 samples (1 second at 16kHz) for memory efficiency

### Common Issues

- CUDA OOM errors: Reduce audio length in `read_preprocess_music()` function in `utils.py:30`
- Qdrant connection issues: Verify `.env` file credentials
- YouTube download failures: Check regional availability and update yt_dlp