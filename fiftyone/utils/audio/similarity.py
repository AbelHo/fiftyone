"""
Audio similarity utilities based on FFT features.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
from typing import List, Optional, Dict, Tuple, Union

import numpy as np

import fiftyone as fo
import fiftyone.core.utils as fou

logger = logging.getLogger(__name__)


def compute_fft_features(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_features: int = 128,
    aggregation: str = "mean",
    normalize: bool = False,
) -> np.ndarray:
    """Compute FFT-based features from audio.

    Args:
        audio: audio data as 1D numpy array
        sample_rate: sample rate in Hz
        n_fft: FFT window size
        hop_length: hop length between frames
        n_features: number of frequency bins to return
        aggregation: how to aggregate across time, one of:
            - "mean": mean across all frames
            - "max": max across all frames
            - "std": standard deviation across frames
            - "all": return all frames (no aggregation)
        normalize: whether to L2-normalize the output features

    Returns:
        feature vector of shape (n_features,) or (n_frames, n_features)
    """
    from scipy import signal

    # Ensure audio is 1D
    if len(audio.shape) > 1:
        audio = np.mean(
            audio, axis=0 if audio.shape[0] > audio.shape[1] else 1
        )

    # Compute STFT
    window = signal.windows.hann(n_fft)
    freqs, times, Zxx = signal.stft(
        audio,
        fs=sample_rate,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        return_onesided=True,
    )

    # Compute magnitude spectrum
    magnitude = np.abs(Zxx)

    # Reduce to n_features frequency bins
    if magnitude.shape[0] > n_features:
        # Bin frequencies
        bin_size = magnitude.shape[0] // n_features
        binned = np.zeros((n_features, magnitude.shape[1]))
        for i in range(n_features):
            start = i * bin_size
            end = (
                start + bin_size if i < n_features - 1 else magnitude.shape[0]
            )
            binned[i] = np.mean(magnitude[start:end], axis=0)
        magnitude = binned
    elif magnitude.shape[0] < n_features:
        # Interpolate
        from scipy import interpolate

        x_old = np.linspace(0, 1, magnitude.shape[0])
        x_new = np.linspace(0, 1, n_features)
        f = interpolate.interp1d(x_old, magnitude, axis=0, kind="linear")
        magnitude = f(x_new)

    # Convert to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    # Aggregate across time
    if aggregation == "mean":
        features = np.mean(magnitude_db, axis=1)
    elif aggregation == "max":
        features = np.max(magnitude_db, axis=1)
    elif aggregation == "std":
        features = np.std(magnitude_db, axis=1)
    elif aggregation == "all":
        features = magnitude_db.T  # (n_frames, n_features)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    # Normalize if requested
    if normalize:
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

    return features


def compute_mean_fft(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    normalize: bool = True,
) -> np.ndarray:
    """Compute mean FFT spectrum of audio.

    This is a simple feature that represents the average spectral content
    of the audio signal.

    Args:
        audio: audio data as 1D numpy array
        sample_rate: sample rate in Hz
        n_fft: FFT window size
        normalize: whether to normalize the output

    Returns:
        mean FFT magnitude spectrum of shape (n_fft // 2 + 1,)
    """
    from scipy import signal

    # Ensure audio is 1D
    if len(audio.shape) > 1:
        audio = np.mean(
            audio, axis=0 if audio.shape[0] > audio.shape[1] else 1
        )

    # Compute STFT
    window = signal.windows.hann(n_fft)
    _, _, Zxx = signal.stft(
        audio,
        fs=sample_rate,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft // 2,
        nfft=n_fft,
        return_onesided=True,
    )

    # Compute mean magnitude
    magnitude = np.abs(Zxx)
    mean_spectrum = np.mean(magnitude, axis=1)

    if normalize:
        norm = np.linalg.norm(mean_spectrum)
        if norm > 0:
            mean_spectrum = mean_spectrum / norm

    return mean_spectrum


def compute_similarity(
    features1: np.ndarray,
    features2: np.ndarray,
    method: str = "cosine",
) -> float:
    """Compute similarity between two feature vectors.

    Args:
        features1: first feature vector
        features2: second feature vector
        method: similarity method, one of:
            - "cosine": cosine similarity
            - "euclidean": negative euclidean distance
            - "correlation": Pearson correlation coefficient
            - "dot": dot product

    Returns:
        similarity score (higher = more similar)
    """
    if method == "cosine":
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(features1, features2) / (norm1 * norm2))

    elif method == "euclidean":
        # Return negative distance so higher = more similar
        return -float(np.linalg.norm(features1 - features2))

    elif method == "correlation":
        if np.std(features1) == 0 or np.std(features2) == 0:
            return 0.0
        return float(np.corrcoef(features1, features2)[0, 1])

    elif method == "dot":
        return float(np.dot(features1, features2))

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def compute_pairwise_similarity(
    features_list: List[np.ndarray],
    method: str = "cosine",
) -> np.ndarray:
    """Compute pairwise similarity matrix.

    Args:
        features_list: list of feature vectors
        method: similarity method

    Returns:
        similarity matrix of shape (n, n)
    """
    n = len(features_list)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            sim = compute_similarity(
                features_list[i], features_list[j], method
            )
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    return similarity_matrix


def compute_dataset_fft_features(
    dataset: fo.Dataset,
    audio_path_field: str = "audio_path",
    output_field: str = "fft_features",
    n_fft: int = 2048,
    n_features: int = 128,
    aggregation: str = "mean",
    progress: bool = True,
) -> None:
    """Compute FFT features for all samples in a dataset.

    Args:
        dataset: the dataset to process
        audio_path_field: field containing audio file paths
        output_field: field to store computed features
        n_fft: FFT window size
        n_features: number of frequency bins
        aggregation: aggregation method
        progress: whether to show progress bar
    """
    from .core import load_audio

    samples = list(dataset)
    with fou.ProgressBar(total=len(samples), progress=progress) as pb:
        for sample in pb(samples):
            if not sample.has_field(audio_path_field):
                continue
            audio_path = sample[audio_path_field]
            if audio_path is None:
                continue

            try:
                audio, sr = load_audio(audio_path)
                features = compute_fft_features(
                    audio,
                    sr,
                    n_fft=n_fft,
                    n_features=n_features,
                    aggregation=aggregation,
                )
                sample[output_field] = features.tolist()
                sample.save()

            except Exception as e:
                logger.warning(
                    f"Failed to compute features for {audio_path}: {e}"
                )


def compute_dataset_similarity(
    dataset: fo.Dataset,
    reference_features: Optional[np.ndarray] = None,
    reference_sample_id: Optional[str] = None,
    features_field: str = "fft_features",
    output_field: str = "similarity",
    method: str = "cosine",
    progress: bool = True,
) -> None:
    """Compute similarity scores for all samples against a reference.

    If neither reference_features nor reference_sample_id is provided,
    computes similarity against the mean features of the dataset.

    Args:
        dataset: the dataset to process
        reference_features: optional reference feature vector
        reference_sample_id: optional sample ID to use as reference
        features_field: field containing feature vectors
        output_field: field to store similarity scores
        method: similarity method
        progress: whether to show progress bar
    """
    # Get reference features
    if reference_features is None:
        if reference_sample_id is not None:
            sample = dataset[reference_sample_id]
            reference_features = np.array(sample[features_field])
        else:
            # Compute mean features
            all_features = []
            for sample in dataset:
                if sample.has_field(features_field):
                    features = sample[features_field]
                    if features is not None:
                        all_features.append(np.array(features))

            if not all_features:
                raise ValueError("No samples have computed features")

            reference_features = np.mean(all_features, axis=0)

    # Compute similarities
    samples = list(dataset)
    with fou.ProgressBar(total=len(samples), progress=progress) as pb:
        for sample in pb(samples):
            if not sample.has_field(features_field):
                continue
            features = sample[features_field]
            if features is None:
                continue

            features = np.array(features)
            sim = compute_similarity(features, reference_features, method)
            sample[output_field] = sim
            sample.save()


def find_similar_samples(
    dataset: fo.Dataset,
    query_features: np.ndarray,
    features_field: str = "fft_features",
    method: str = "cosine",
    k: int = 10,
) -> List[Tuple[str, float]]:
    """Find the k most similar samples to a query.

    Args:
        dataset: the dataset to search
        query_features: query feature vector
        features_field: field containing feature vectors
        method: similarity method
        k: number of results to return

    Returns:
        list of (sample_id, similarity) tuples, sorted by similarity
    """
    results = []

    for sample in dataset:
        if not sample.has_field(features_field):
            continue
        features = sample[features_field]
        if features is None:
            continue

        features = np.array(features)
        sim = compute_similarity(features, query_features, method)
        results.append((sample.id, sim))

    # Sort by similarity (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:k]


def create_similarity_index(
    dataset: fo.Dataset,
    features_field: str = "fft_features",
    brain_key: str = "audio_similarity",
) -> None:
    """Create a similarity index for efficient nearest neighbor search.

    This creates a FiftyOne brain similarity index that enables fast
    similarity queries.

    Args:
        dataset: the dataset to index
        features_field: field containing feature vectors
        brain_key: key for the brain run
    """
    import fiftyone.brain as fob

    # Get embeddings from dataset
    embeddings = []
    sample_ids = []

    for sample in dataset:
        if sample.has_field(features_field):
            features = sample[features_field]
            if features is not None:
                embeddings.append(np.array(features))
                sample_ids.append(sample.id)

    if not embeddings:
        raise ValueError("No samples have computed features")

    embeddings = np.array(embeddings)

    # Compute similarity index
    fob.compute_similarity(
        dataset,
        embeddings=embeddings,
        brain_key=brain_key,
    )

    logger.info(f"Created similarity index with {len(embeddings)} samples")
