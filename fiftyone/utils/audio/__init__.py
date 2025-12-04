"""
FiftyOne audio processing utilities.

This module provides utilities for working with audio data in FiftyOne,
including spectrogram generation, audio metadata, and dataset creation.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from .core import (
    load_audio,
    save_audio,
    get_audio_info,
    normalize_audio,
)
from .spectrogram import (
    SpectrogramConfig,
    generate_spectrogram,
    generate_spectrogram_image,
    save_spectrogram,
)
from .metadata import AudioMetadata
from .dataset import (
    create_audio_dataset,
    add_audio_samples,
    generate_spectrograms_for_dataset,
)
from .detection import (
    AudioDetector,
    DetectionResult,
    ThresholdDetector,
    SpectralPeakDetector,
    run_detection,
    create_custom_detector,
)
from .similarity import (
    compute_fft_features,
    compute_mean_fft,
    compute_similarity,
    compute_pairwise_similarity,
    compute_dataset_fft_features,
    compute_dataset_similarity,
    find_similar_samples,
)
from .generator import (
    generate_sine_wave,
    generate_chirp,
    generate_noise,
    generate_tone_burst,
    generate_harmonic_tone,
    generate_click,
    generate_test_audio_suite,
)

__all__ = [
    # Core
    "load_audio",
    "save_audio",
    "get_audio_info",
    "normalize_audio",
    # Spectrogram
    "SpectrogramConfig",
    "generate_spectrogram",
    "generate_spectrogram_image",
    "save_spectrogram",
    # Metadata
    "AudioMetadata",
    # Dataset
    "create_audio_dataset",
    "add_audio_samples",
    "generate_spectrograms_for_dataset",
    # Detection
    "AudioDetector",
    "DetectionResult",
    "ThresholdDetector",
    "SpectralPeakDetector",
    "run_detection",
    "create_custom_detector",
    # Similarity
    "compute_fft_features",
    "compute_mean_fft",
    "compute_similarity",
    "compute_pairwise_similarity",
    "compute_dataset_fft_features",
    "compute_dataset_similarity",
    "find_similar_samples",
    # Generator
    "generate_sine_wave",
    "generate_chirp",
    "generate_noise",
    "generate_tone_burst",
    "generate_harmonic_tone",
    "generate_click",
    "generate_test_audio_suite",
]
