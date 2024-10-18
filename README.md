# Music DB

This project was created as an exercise to experiment with Vector databases.

## Project Overview

The aim was to create meaningful embeddings of songs and to store them in a vector database. This database can then be queried by embeddings of a desired song, after which you recieve similar songs that are nearby in the embedding space. The embeddings were generated from **Pretrained Audio Neural Networks** or PANNs. The PANNs implementation in pytorch was used which can be found [here](https://github.com/qiuqiangkong/panns_inference).

## Files and Directories

- **`add_music_embeddings.py`**: Iterates over the music in the specified folder, generates embeddings and inserts the same into a new or existing collection in your Quadrant DB cluster.
- **`query_database.py`**: Queries an input vector of a selected song in database to produce `n` similar songs.Current functionality restricted to one song at a time, read from a specified folder. Future work will be done to extend its capability to work more as a more robust recommendation engine.
- **`requirements.yaml`**: Use this file to replicate the conda environment if needed.
- **`utils.py`**: Contains helper functions.
- **`music_download.py`**: Used for downloading music from a specified playlist URL.

