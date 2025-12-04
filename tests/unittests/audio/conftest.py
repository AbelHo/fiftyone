"""
Audio processing unit tests configuration.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import tempfile
import pytest
import numpy as np


@pytest.fixture(scope="session")
def test_audio_dir():
    """Create a temporary directory with test audio files."""
    from fiftyone.utils.audio import generate_test_audio_suite

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_files = generate_test_audio_suite(tmpdir)
        yield tmpdir, audio_files


@pytest.fixture
def simple_audio():
    """Generate a simple test audio signal."""
    sample_rate = 44100
    duration = 1.0
    frequency = 440

    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    return audio, sample_rate


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_audio_file(temp_dir, simple_audio):
    """Create a temporary audio file."""
    from fiftyone.utils.audio import save_audio

    audio, sr = simple_audio
    filepath = os.path.join(temp_dir, "test_audio.wav")
    save_audio(audio, filepath, sr)
    return filepath
