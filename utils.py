import numpy as np
import librosa
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast


device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_music(file_path, n_mfcc=50):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = librosa.util.normalize(mfcc)
    return np.expand_dims(mfcc.T, axis=0)  # Transpose to shape (timesteps, features)

def pad_and_truncate_sequences(sequences, maxlen=5000, padding_value=0.0) -> torch.Tensor:
    """
    Pads and truncates sequences to the specified maximum length.

    Args:
        sequences (list of tensors): List of variable-length sequences (each is a 1D tensor).
        maxlen (int): Maximum length to pad or truncate to.
        padding_value (float): Value used for padding (default is 0.0).

    Returns:
        Tensor: Padded and truncated tensor with shape (batch_size, maxlen, feature_dim).
    """
    sequences = torch.tensor(sequences, dtype=torch.float32)
    # Convert list of sequences to a batch tensor using pad_sequence
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)

    # Truncate sequences if they exceed maxlen
    if padded_sequences.size(1) > maxlen:
        padded_sequences = padded_sequences[:, :maxlen, :]
    # Pad sequences to maxlen if they are shorter
    elif padded_sequences.size(1) < maxlen:
        # Create a tensor of shape (batch_size, maxlen, feature_dim) filled with padding_value
        pad_shape = (padded_sequences.size(0), maxlen - padded_sequences.size(1), padded_sequences.size(2))
        padding = torch.full(pad_shape, padding_value)
        padded_sequences = torch.cat([padded_sequences, padding], dim=1)

    return padded_sequences


def create_music_embeddings(mfccs: torch.Tensor, model) -> np.ndarray:
    mfccs = mfccs.to(device)
    model.eval()

    with torch.no_grad(),autocast(dtype=torch.float32):
        embeddings = model.encoder(mfccs)
    embeddings = embeddings.squeeze(0)
    embeddings = torch.mean(embeddings, dim=0)

    return embeddings.detach().cpu().numpy()
