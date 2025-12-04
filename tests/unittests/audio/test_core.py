"""
Unit tests for audio core functionality.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
# pylint: disable=E1101
import os
import tempfile

import numpy as np
import pytest


class TestLoadAudio:
    """Tests for load_audio function."""

    def test_load_wav(self, temp_audio_file):
        """Test loading a WAV file."""
        from fiftyone.utils.audio import load_audio

        audio, sr = load_audio(temp_audio_file)

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == 44100
        assert len(audio) > 0

    def test_load_audio_mono(self, temp_audio_file):
        """Test loading audio as mono."""
        from fiftyone.utils.audio import load_audio

        audio, sr = load_audio(temp_audio_file, mono=True)

        assert len(audio.shape) == 1

    def test_load_audio_with_offset(self, temp_audio_file):
        """Test loading audio with offset."""
        from fiftyone.utils.audio import load_audio

        audio_full, sr = load_audio(temp_audio_file)
        audio_offset, _ = load_audio(temp_audio_file, offset=0.5)

        # Audio with offset should be shorter
        assert len(audio_offset) < len(audio_full)

    def test_load_audio_with_duration(self, temp_audio_file):
        """Test loading audio with duration limit."""
        from fiftyone.utils.audio import load_audio

        audio, sr = load_audio(temp_audio_file, duration=0.5)

        expected_samples = int(0.5 * sr)
        assert len(audio) == expected_samples

    def test_load_audio_file_not_found(self):
        """Test loading non-existent file raises error."""
        from fiftyone.utils.audio import load_audio

        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/path/audio.wav")


class TestGetAudioInfo:
    """Tests for get_audio_info function."""

    def test_get_info(self, temp_audio_file):
        """Test getting audio info."""
        from fiftyone.utils.audio import get_audio_info

        info = get_audio_info(temp_audio_file)

        assert "sample_rate" in info
        assert "channels" in info
        assert "duration" in info
        assert "samples" in info
        assert "format" in info
        assert "size_bytes" in info

        assert info["sample_rate"] == 44100
        assert info["channels"] == 1
        assert info["duration"] > 0
        assert info["format"] == "wav"

    def test_get_info_file_not_found(self):
        """Test getting info for non-existent file."""
        from fiftyone.utils.audio import get_audio_info

        with pytest.raises(FileNotFoundError):
            get_audio_info("/nonexistent/path/audio.wav")


class TestNormalizeAudio:
    """Tests for normalize_audio function."""

    def test_peak_normalize(self, simple_audio):
        """Test peak normalization."""
        from fiftyone.utils.audio import normalize_audio

        audio, _ = simple_audio
        normalized = normalize_audio(audio, method="peak", target_level=1.0)

        assert np.max(np.abs(normalized)) == pytest.approx(1.0, rel=1e-5)

    def test_rms_normalize(self, simple_audio):
        """Test RMS normalization."""
        from fiftyone.utils.audio import normalize_audio

        audio, _ = simple_audio
        target_rms = 0.5
        normalized = normalize_audio(
            audio, method="rms", target_level=target_rms
        )

        actual_rms = np.sqrt(np.mean(normalized**2))
        assert actual_rms == pytest.approx(target_rms, rel=1e-2)

    def test_invalid_method(self, simple_audio):
        """Test invalid normalization method."""
        from fiftyone.utils.audio import normalize_audio

        audio, _ = simple_audio
        with pytest.raises(ValueError):
            normalize_audio(audio, method="invalid")


class TestSaveAudio:
    """Tests for save_audio function."""

    def test_save_wav(self, simple_audio, temp_dir):
        """Test saving as WAV."""
        from fiftyone.utils.audio import save_audio, load_audio

        audio, sr = simple_audio
        filepath = os.path.join(temp_dir, "output.wav")

        save_audio(audio, filepath, sr)

        assert os.path.exists(filepath)

        # Verify by loading back
        loaded_audio, loaded_sr = load_audio(filepath)
        assert loaded_sr == sr
        assert len(loaded_audio) == len(audio)

    def test_save_creates_directory(self, simple_audio, temp_dir):
        """Test that save_audio creates directories."""
        from fiftyone.utils.audio import save_audio

        audio, sr = simple_audio
        filepath = os.path.join(temp_dir, "subdir", "nested", "output.wav")

        save_audio(audio, filepath, sr)

        assert os.path.exists(filepath)
