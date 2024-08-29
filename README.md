# Music DB

This project was created as an exercise to experiment with Vector databases.

## Project Overview

The aim was to extract music features from a simple file format (in this case, MP3). After several experiments using functionalities provided in `librosa`, the project was supplemented by training an autoencoder to use as a feature extractor. The trained model's encoder is used to create "music embeddings" that can be added to and queried from the database.

## Files and Directories

- **`qdb_test.py`**: A quick and dirty implementation to add and query embeddings and see the output results.
- **`requirements.yaml`**: Use this file to replicate the conda environment if needed.

## Custom Training and Testing

If you want to train and test on your own set of audio files:

1. Create a directory named `music_files` for training.
2. Create a directory named `inference_music_files` for inference.
