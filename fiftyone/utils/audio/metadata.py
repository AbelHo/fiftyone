"""
Audio metadata utilities.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import os
from typing import Optional

import fiftyone.core.fields as fof
import fiftyone.core.metadata as fom

logger = logging.getLogger(__name__)


class AudioMetadata(fom.Metadata):
    """Class for storing metadata about audio samples.

    Attributes:
        size_bytes: the size of the audio file on disk, in bytes
        mime_type: the MIME type of the audio
        sample_rate: the sample rate in Hz
        channels: the number of audio channels
        duration: the duration of the audio in seconds
        samples: the total number of samples
        bit_depth: the bit depth of the audio
        encoding: the audio encoding format
    """

    sample_rate = fof.IntField()
    channels = fof.IntField()
    duration = fof.FloatField()
    samples = fof.IntField()
    bit_depth = fof.IntField()
    encoding = fof.StringField()


def compute_audio_metadata(filepath: str) -> AudioMetadata:
    """Compute metadata for an audio file.

    Args:
        filepath: path to the audio file

    Returns:
        an :class:`AudioMetadata` instance

    Raises:
        FileNotFoundError: if the audio file does not exist
        ValueError: if the audio file cannot be read
    """
    from . import core

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    # Get basic file info
    size_bytes = os.path.getsize(filepath)

    # Get audio info
    info = core.get_audio_info(filepath)

    # Determine MIME type
    ext = os.path.splitext(filepath)[1].lower().lstrip(".")
    mime_types = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "m4a": "audio/mp4",
        "aac": "audio/aac",
        "wma": "audio/x-ms-wma",
        "aiff": "audio/aiff",
        "au": "audio/basic",
    }
    mime_type = mime_types.get(ext, f"audio/{ext}")

    return AudioMetadata(
        size_bytes=size_bytes,
        mime_type=mime_type,
        sample_rate=info["sample_rate"],
        channels=info["channels"],
        duration=info["duration"],
        samples=info["samples"],
        encoding=info["format"],
    )


def get_audio_metadata(filepath: str) -> AudioMetadata:
    """Get metadata for an audio file.

    This is an alias for :func:`compute_audio_metadata`.

    Args:
        filepath: path to the audio file

    Returns:
        an :class:`AudioMetadata` instance
    """
    return compute_audio_metadata(filepath)
