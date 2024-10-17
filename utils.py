import numpy as np
import librosa
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


def read_preprocess_music(file_path: str) -> np.ndarray:
    (audio, _) = librosa.core.load(file_path)
    audio = audio[None, :]

    return audio
