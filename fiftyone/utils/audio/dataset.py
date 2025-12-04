"""
Audio dataset utilities.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import os
from typing import List, Optional, Union

import numpy as np

import fiftyone as fo
import fiftyone.core.utils as fou

from .core import load_audio, get_audio_info
from .spectrogram import (
    SpectrogramConfig,
    save_spectrogram,
    generate_spectrogram_image,
)
from .metadata import AudioMetadata, compute_audio_metadata

logger = logging.getLogger(__name__)


def create_audio_dataset(
    name: Optional[str] = None,
    audio_paths: Optional[List[str]] = None,
    audio_dir: Optional[str] = None,
    extensions: Optional[List[str]] = None,
    spectrogram_dir: Optional[str] = None,
    spectrogram_config: Optional[SpectrogramConfig] = None,
    generate_spectrograms: bool = True,
    compute_metadata: bool = True,
    persistent: bool = False,
    tags: Optional[List[str]] = None,
    progress: bool = True,
) -> fo.Dataset:
    """Create a FiftyOne dataset from audio files.

    This function creates a dataset where each audio file is represented as a
    sample with:
    - The spectrogram image as the primary media (for visualization)
    - A reference to the original audio file
    - Audio metadata
    - Spectrogram configuration used

    Args:
        name: optional name for the dataset
        audio_paths: list of paths to audio files
        audio_dir: directory containing audio files (alternative to audio_paths)
        extensions: list of audio file extensions to include (default: common audio formats)
        spectrogram_dir: directory to save spectrogram images. If None, creates
            a subdirectory next to audio files
        spectrogram_config: configuration for spectrogram generation
        generate_spectrograms: whether to generate spectrogram images
        compute_metadata: whether to compute audio metadata
        persistent: whether the dataset should persist
        tags: optional list of tags to add to all samples
        progress: whether to show progress bar

    Returns:
        a :class:`fiftyone.core.dataset.Dataset`

    Raises:
        ValueError: if neither audio_paths nor audio_dir is provided
    """
    if audio_paths is None and audio_dir is None:
        raise ValueError("Either audio_paths or audio_dir must be provided")

    # Default extensions
    if extensions is None:
        extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff"]

    # Collect audio files
    if audio_paths is None:
        audio_paths = []
        for root, _, files in os.walk(audio_dir):
            for f in files:
                if any(f.lower().endswith(ext) for ext in extensions):
                    audio_paths.append(os.path.join(root, f))

    if not audio_paths:
        raise ValueError("No audio files found")

    logger.info(f"Found {len(audio_paths)} audio files")

    # Create dataset
    dataset = fo.Dataset(name=name, persistent=persistent)

    # Add samples
    add_audio_samples(
        dataset,
        audio_paths=audio_paths,
        spectrogram_dir=spectrogram_dir,
        spectrogram_config=spectrogram_config,
        generate_spectrograms=generate_spectrograms,
        compute_metadata=compute_metadata,
        tags=tags,
        progress=progress,
    )

    return dataset


def add_audio_samples(
    dataset: fo.Dataset,
    audio_paths: List[str],
    spectrogram_dir: Optional[str] = None,
    spectrogram_config: Optional[SpectrogramConfig] = None,
    generate_spectrograms: bool = True,
    compute_metadata: bool = True,
    tags: Optional[List[str]] = None,
    progress: bool = True,
) -> None:
    """Add audio samples to an existing dataset.

    Args:
        dataset: the dataset to add samples to
        audio_paths: list of paths to audio files
        spectrogram_dir: directory to save spectrogram images
        spectrogram_config: configuration for spectrogram generation
        generate_spectrograms: whether to generate spectrogram images
        compute_metadata: whether to compute audio metadata
        tags: optional list of tags to add to samples
        progress: whether to show progress bar
    """
    if spectrogram_config is None:
        spectrogram_config = SpectrogramConfig()

    samples = []

    with fou.ProgressBar(total=len(audio_paths), progress=progress) as pb:
        for audio_path in pb(audio_paths):
            try:
                sample = _create_audio_sample(
                    audio_path,
                    spectrogram_dir=spectrogram_dir,
                    spectrogram_config=spectrogram_config,
                    generate_spectrogram=generate_spectrograms,
                    compute_metadata=compute_metadata,
                    tags=tags,
                )
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {e}")
                continue

    dataset.add_samples(samples)
    logger.info(f"Added {len(samples)} audio samples to dataset")


def _create_audio_sample(
    audio_path: str,
    spectrogram_dir: Optional[str] = None,
    spectrogram_config: Optional[SpectrogramConfig] = None,
    generate_spectrogram: bool = True,
    compute_metadata: bool = True,
    tags: Optional[List[str]] = None,
) -> fo.Sample:
    """Create a single audio sample.

    Args:
        audio_path: path to the audio file
        spectrogram_dir: directory to save spectrogram image
        spectrogram_config: configuration for spectrogram generation
        generate_spectrogram: whether to generate spectrogram image
        compute_metadata: whether to compute audio metadata
        tags: optional list of tags

    Returns:
        a :class:`fiftyone.core.sample.Sample`
    """
    if spectrogram_config is None:
        spectrogram_config = SpectrogramConfig()

    audio_path = os.path.abspath(audio_path)
    audio_filename = os.path.basename(audio_path)
    audio_name = os.path.splitext(audio_filename)[0]

    # Determine spectrogram path
    if spectrogram_dir is None:
        spectrogram_dir = os.path.join(
            os.path.dirname(audio_path), "spectrograms"
        )

    os.makedirs(spectrogram_dir, exist_ok=True)
    spectrogram_path = os.path.join(
        spectrogram_dir, f"{audio_name}_spectrogram.png"
    )

    # Generate spectrogram if needed
    if generate_spectrogram:
        audio, sr = load_audio(audio_path)
        save_spectrogram(
            audio,
            sr,
            spectrogram_path,
            config=spectrogram_config,
            show_colorbar=False,
            show_axes=False,
        )

    # Create sample with spectrogram as primary media
    # This allows the spectrogram to be displayed in the UI
    if generate_spectrogram and os.path.exists(spectrogram_path):
        sample = fo.Sample(filepath=spectrogram_path)
    else:
        # If no spectrogram, use audio file with custom media type
        sample = fo.Sample(filepath=audio_path, media_type="audio")

    # Store reference to original audio file
    sample["audio_path"] = audio_path

    # Store spectrogram configuration
    sample["spectrogram_config"] = spectrogram_config.to_dict()

    # Compute and store audio metadata
    if compute_metadata:
        try:
            audio_meta = compute_audio_metadata(audio_path)
            sample["audio_metadata"] = {
                "sample_rate": audio_meta.sample_rate,
                "channels": audio_meta.channels,
                "duration": audio_meta.duration,
                "samples": audio_meta.samples,
                "encoding": audio_meta.encoding,
                "size_bytes": audio_meta.size_bytes,
            }
        except Exception as e:
            logger.warning(f"Failed to compute metadata for {audio_path}: {e}")

    # Add tags
    if tags:
        sample.tags = list(tags)

    return sample


def generate_spectrograms_for_dataset(
    dataset: fo.Dataset,
    spectrogram_dir: Optional[str] = None,
    spectrogram_config: Optional[SpectrogramConfig] = None,
    audio_path_field: str = "audio_path",
    output_field: str = "spectrogram_path",
    overwrite: bool = False,
    progress: bool = True,
) -> None:
    """Generate spectrogram images for all samples in a dataset.

    This is useful for regenerating spectrograms with different parameters.

    Args:
        dataset: the dataset to process
        spectrogram_dir: directory to save spectrogram images
        spectrogram_config: configuration for spectrogram generation
        audio_path_field: field containing path to audio file
        output_field: field to store path to spectrogram image
        overwrite: whether to overwrite existing spectrograms
        progress: whether to show progress bar
    """
    if spectrogram_config is None:
        spectrogram_config = SpectrogramConfig()

    samples = list(dataset)
    with fou.ProgressBar(total=len(samples), progress=progress) as pb:
        for sample in pb(samples):
            audio_path = sample[audio_path_field]
            if audio_path is None:
                continue

            # Determine spectrogram path
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            if spectrogram_dir is None:
                spec_dir = os.path.join(
                    os.path.dirname(audio_path), "spectrograms"
                )
            else:
                spec_dir = spectrogram_dir

            os.makedirs(spec_dir, exist_ok=True)
            spectrogram_path = os.path.join(
                spec_dir, f"{audio_name}_spectrogram.png"
            )

            # Skip if exists and not overwriting
            if os.path.exists(spectrogram_path) and not overwrite:
                sample[output_field] = spectrogram_path
                sample.save()
                continue

            try:
                # Load audio and generate spectrogram
                audio, sr = load_audio(audio_path)
                save_spectrogram(
                    audio,
                    sr,
                    spectrogram_path,
                    config=spectrogram_config,
                    show_colorbar=False,
                    show_axes=False,
                )

                sample[output_field] = spectrogram_path
                sample["spectrogram_config"] = spectrogram_config.to_dict()
                sample.save()

            except Exception as e:
                logger.warning(
                    f"Failed to generate spectrogram for {audio_path}: {e}"
                )


def update_dataset_filepaths_to_spectrograms(
    dataset: fo.Dataset,
    spectrogram_path_field: str = "spectrogram_path",
    backup_field: str = "original_filepath",
) -> None:
    """Update dataset filepaths to point to spectrogram images.

    This allows spectrograms to be displayed as the primary media in the UI.

    Args:
        dataset: the dataset to update
        spectrogram_path_field: field containing spectrogram paths
        backup_field: field to store original filepath
    """
    for sample in dataset:
        spectrogram_path = sample.get(spectrogram_path_field)
        if spectrogram_path and os.path.exists(spectrogram_path):
            # Backup original filepath
            sample[backup_field] = sample.filepath
            # Update filepath to spectrogram
            sample.filepath = spectrogram_path
            sample.save()
