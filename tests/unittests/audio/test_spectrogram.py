"""
Unit tests for spectrogram generation.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import tempfile

import numpy as np
import pytest


class TestSpectrogramConfig:
    """Tests for SpectrogramConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        from fiftyone.utils.audio import SpectrogramConfig

        config = SpectrogramConfig()

        assert config.n_fft == 2048
        assert config.hop_length == 512
        assert config.window == "hann"
        assert config.colormap == "viridis"

    def test_custom_config(self):
        """Test custom configuration."""
        from fiftyone.utils.audio import SpectrogramConfig

        config = SpectrogramConfig(
            n_fft=4096,
            hop_length=1024,
            colormap="plasma",
            mel_scale=True,
            n_mels=64,
        )

        assert config.n_fft == 4096
        assert config.hop_length == 1024
        assert config.colormap == "plasma"
        assert config.mel_scale is True
        assert config.n_mels == 64

    def test_overlap_property(self):
        """Test overlap calculation."""
        from fiftyone.utils.audio import SpectrogramConfig

        config = SpectrogramConfig(n_fft=2048, hop_length=512)

        assert config.overlap == 2048 - 512
        assert config.overlap_ratio == pytest.approx(0.75)

    def test_invalid_n_fft(self):
        """Test invalid n_fft raises error."""
        from fiftyone.utils.audio import SpectrogramConfig

        with pytest.raises(ValueError):
            SpectrogramConfig(n_fft=0)

        with pytest.raises(ValueError):
            SpectrogramConfig(n_fft=-1)

    def test_invalid_window(self):
        """Test invalid window raises error."""
        from fiftyone.utils.audio import SpectrogramConfig

        with pytest.raises(ValueError):
            SpectrogramConfig(window="invalid_window")

    def test_to_dict(self):
        """Test serialization to dict."""
        from fiftyone.utils.audio import SpectrogramConfig

        config = SpectrogramConfig(n_fft=4096, colormap="plasma")
        d = config.to_dict()

        assert d["n_fft"] == 4096
        assert d["colormap"] == "plasma"

    def test_from_dict(self):
        """Test deserialization from dict."""
        from fiftyone.utils.audio import SpectrogramConfig

        d = {"n_fft": 4096, "hop_length": 1024, "colormap": "inferno"}
        config = SpectrogramConfig.from_dict(d)

        assert config.n_fft == 4096
        assert config.hop_length == 1024
        assert config.colormap == "inferno"


class TestGenerateSpectrogram:
    """Tests for generate_spectrogram function."""

    def test_basic_spectrogram(self, simple_audio):
        """Test basic spectrogram generation."""
        from fiftyone.utils.audio import generate_spectrogram

        audio, sr = simple_audio
        spec, freqs, times = generate_spectrogram(audio, sr)

        assert isinstance(spec, np.ndarray)
        assert len(spec.shape) == 2
        assert len(freqs) == spec.shape[0]
        assert len(times) == spec.shape[1]

    def test_spectrogram_shape(self, simple_audio):
        """Test spectrogram shape with different parameters."""
        from fiftyone.utils.audio import (
            generate_spectrogram,
            SpectrogramConfig,
        )

        audio, sr = simple_audio

        # Test with different n_fft
        config1 = SpectrogramConfig(n_fft=1024)
        spec1, freqs1, _ = generate_spectrogram(audio, sr, config1)

        config2 = SpectrogramConfig(n_fft=4096)
        spec2, freqs2, _ = generate_spectrogram(audio, sr, config2)

        # Larger n_fft should give more frequency bins
        assert len(freqs2) > len(freqs1)

    def test_spectrogram_frequency_limits(self, simple_audio):
        """Test spectrogram with frequency limits."""
        from fiftyone.utils.audio import (
            generate_spectrogram,
            SpectrogramConfig,
        )

        audio, sr = simple_audio

        config = SpectrogramConfig(freq_min=100, freq_max=5000)
        spec, freqs, times = generate_spectrogram(audio, sr, config)

        assert freqs[0] >= 100
        assert freqs[-1] <= 5000

    def test_mel_spectrogram(self, simple_audio):
        """Test mel spectrogram generation."""
        from fiftyone.utils.audio import (
            generate_spectrogram,
            SpectrogramConfig,
        )

        audio, sr = simple_audio

        config = SpectrogramConfig(mel_scale=True, n_mels=64)
        spec, freqs, times = generate_spectrogram(audio, sr, config)

        assert spec.shape[0] == 64  # n_mels

    def test_normalized_spectrogram(self, simple_audio):
        """Test normalized spectrogram values."""
        from fiftyone.utils.audio import (
            generate_spectrogram,
            SpectrogramConfig,
        )

        audio, sr = simple_audio

        config = SpectrogramConfig(normalize=True)
        spec, _, _ = generate_spectrogram(audio, sr, config)

        # Normalized values should be in [0, 1]
        assert np.min(spec) >= 0
        assert np.max(spec) <= 1


class TestGenerateSpectrogramImage:
    """Tests for generate_spectrogram_image function."""

    def test_image_generation(self, simple_audio):
        """Test spectrogram image generation."""
        from fiftyone.utils.audio import generate_spectrogram_image

        audio, sr = simple_audio
        img = generate_spectrogram_image(audio, sr)

        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 3
        assert img.shape[2] == 3  # RGB

    def test_image_save(self, simple_audio, temp_dir):
        """Test saving spectrogram image."""
        from fiftyone.utils.audio import generate_spectrogram_image

        audio, sr = simple_audio
        output_path = os.path.join(temp_dir, "spectrogram.png")

        img = generate_spectrogram_image(audio, sr, output_path=output_path)

        assert os.path.exists(output_path)

    def test_image_dimensions(self, simple_audio):
        """Test spectrogram image dimensions match config."""
        from fiftyone.utils.audio import (
            generate_spectrogram_image,
            SpectrogramConfig,
        )

        audio, sr = simple_audio

        config = SpectrogramConfig(figsize=(8, 4), dpi=50)
        img = generate_spectrogram_image(audio, sr, config=config)

        # Image dimensions should be approximately figsize * dpi
        expected_width = int(8 * 50)
        expected_height = int(4 * 50)

        # Allow some tolerance for axis labels etc.
        assert abs(img.shape[1] - expected_width) < expected_width * 0.3
        assert abs(img.shape[0] - expected_height) < expected_height * 0.3


class TestSaveSpectrogram:
    """Tests for save_spectrogram function."""

    def test_save_basic(self, simple_audio, temp_dir):
        """Test basic spectrogram save."""
        from fiftyone.utils.audio import save_spectrogram

        audio, sr = simple_audio
        output_path = os.path.join(temp_dir, "spec.png")

        result_path = save_spectrogram(audio, sr, output_path)

        assert result_path == output_path
        assert os.path.exists(output_path)

    def test_save_creates_directory(self, simple_audio, temp_dir):
        """Test save creates directories."""
        from fiftyone.utils.audio import save_spectrogram

        audio, sr = simple_audio
        output_path = os.path.join(temp_dir, "nested", "dir", "spec.png")

        save_spectrogram(audio, sr, output_path)

        assert os.path.exists(output_path)


class TestCoordinateConversion:
    """Tests for coordinate conversion functions."""

    def test_spectrogram_to_audio_coords(self):
        """Test converting spectrogram coords to audio coords."""
        from fiftyone.utils.audio.spectrogram import (
            spectrogram_to_audio_coords,
        )

        # Test center of spectrogram
        time, freq = spectrogram_to_audio_coords(
            x=0.5,
            y=0.5,
            spec_shape=(128, 256),
            duration=2.0,
            sample_rate=44100,
            freq_min=0,
            freq_max=22050,
        )

        assert time == pytest.approx(1.0)  # 50% of 2 seconds
        assert freq == pytest.approx(11025)  # 50% of Nyquist

    def test_audio_coords_to_spectrogram(self):
        """Test converting audio coords to spectrogram coords."""
        from fiftyone.utils.audio.spectrogram import (
            audio_coords_to_spectrogram,
        )

        x, y = audio_coords_to_spectrogram(
            time_seconds=1.0,
            frequency_hz=11025,
            duration=2.0,
            sample_rate=44100,
            freq_min=0,
            freq_max=22050,
        )

        assert x == pytest.approx(0.5)
        assert y == pytest.approx(0.5)

    def test_round_trip_conversion(self):
        """Test round-trip coordinate conversion."""
        from fiftyone.utils.audio.spectrogram import (
            spectrogram_to_audio_coords,
            audio_coords_to_spectrogram,
        )

        original_x, original_y = 0.3, 0.7

        time, freq = spectrogram_to_audio_coords(
            x=original_x,
            y=original_y,
            spec_shape=(128, 256),
            duration=2.0,
            sample_rate=44100,
        )

        recovered_x, recovered_y = audio_coords_to_spectrogram(
            time_seconds=time,
            frequency_hz=freq,
            duration=2.0,
            sample_rate=44100,
        )

        assert recovered_x == pytest.approx(original_x)
        assert recovered_y == pytest.approx(original_y)
