"""
Unit tests for audio generator.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import numpy as np

# pylint: disable=E1101
import os
import pytest


class TestGenerateSineWave:
    """Tests for generate_sine_wave function."""

    def test_basic_sine(self):
        """Test basic sine wave generation."""
        from fiftyone.utils.audio import generate_sine_wave

        audio, sr = generate_sine_wave(
            frequency=440,
            duration=1.0,
            sample_rate=44100,
            amplitude=0.5,
        )

        assert isinstance(audio, np.ndarray)
        assert sr == 44100
        assert len(audio) == 44100
        assert audio.dtype == np.float32
        assert np.max(np.abs(audio)) <= 0.5 + 0.01

    def test_different_frequencies(self):
        """Test different frequencies produce different signals."""
        from fiftyone.utils.audio import generate_sine_wave

        audio1, _ = generate_sine_wave(frequency=440, duration=1.0)
        audio2, _ = generate_sine_wave(frequency=880, duration=1.0)

        assert not np.allclose(audio1, audio2)

    def test_amplitude(self):
        """Test amplitude control."""
        from fiftyone.utils.audio import generate_sine_wave

        audio1, _ = generate_sine_wave(
            frequency=440, duration=1.0, amplitude=0.2
        )
        audio2, _ = generate_sine_wave(
            frequency=440, duration=1.0, amplitude=0.8
        )

        assert np.max(np.abs(audio1)) < np.max(np.abs(audio2))


class TestGenerateChirp:
    """Tests for generate_chirp function."""

    def test_basic_chirp(self):
        """Test basic chirp generation."""
        from fiftyone.utils.audio import generate_chirp

        audio, sr = generate_chirp(
            freq_start=200,
            freq_end=2000,
            duration=1.0,
            sample_rate=44100,
        )

        assert isinstance(audio, np.ndarray)
        assert sr == 44100
        assert len(audio) == 44100

    def test_chirp_range(self):
        """Test chirp frequency range."""
        from fiftyone.utils.audio import generate_chirp

        audio, sr = generate_chirp(
            freq_start=100,
            freq_end=10000,
            duration=2.0,
        )

        # Should have audio for full duration
        assert len(audio) == sr * 2


class TestGenerateNoise:
    """Tests for generate_noise function."""

    def test_white_noise(self):
        """Test white noise generation."""
        from fiftyone.utils.audio import generate_noise

        audio, sr = generate_noise(
            noise_type="white",
            duration=1.0,
            sample_rate=44100,
        )

        assert isinstance(audio, np.ndarray)
        assert sr == 44100
        assert len(audio) == 44100

    def test_pink_noise(self):
        """Test pink noise generation."""
        from fiftyone.utils.audio import generate_noise

        audio, sr = generate_noise(
            noise_type="pink",
            duration=1.0,
        )

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_noise_amplitude(self):
        """Test noise amplitude control."""
        from fiftyone.utils.audio import generate_noise

        audio1, _ = generate_noise(duration=1.0, amplitude=0.1)
        audio2, _ = generate_noise(duration=1.0, amplitude=0.5)

        # Higher amplitude should have higher RMS
        rms1 = np.sqrt(np.mean(audio1**2))
        rms2 = np.sqrt(np.mean(audio2**2))
        assert rms1 < rms2


class TestGenerateToneBurst:
    """Tests for generate_tone_burst function."""

    def test_basic_burst(self):
        """Test basic tone burst generation."""
        from fiftyone.utils.audio import generate_tone_burst

        audio, sr = generate_tone_burst(
            frequency=1000,
            burst_duration=0.1,
            total_duration=1.0,
            onset_time=0.5,
        )

        assert len(audio) == sr

    def test_burst_location(self):
        """Test burst appears at correct location."""
        from fiftyone.utils.audio import generate_tone_burst

        audio, sr = generate_tone_burst(
            frequency=1000,
            burst_duration=0.1,
            total_duration=1.0,
            onset_time=0.5,
        )

        # Check energy is concentrated around 0.5s
        mid_start = int(0.45 * sr)
        mid_end = int(0.65 * sr)
        start_region = audio[:mid_start]
        mid_region = audio[mid_start:mid_end]

        mid_energy = np.sum(mid_region**2)
        start_energy = np.sum(start_region**2)

        assert mid_energy > start_energy


class TestGenerateHarmonicTone:
    """Tests for generate_harmonic_tone function."""

    def test_basic_harmonic(self):
        """Test basic harmonic tone generation."""
        from fiftyone.utils.audio import generate_harmonic_tone

        audio, sr = generate_harmonic_tone(
            fundamental=440,
            duration=1.0,
            n_harmonics=3,
        )

        assert isinstance(audio, np.ndarray)
        assert len(audio) == sr

    def test_multiple_harmonics(self):
        """Test that multiple harmonics create richer sound."""
        from fiftyone.utils.audio import generate_harmonic_tone

        audio1, _ = generate_harmonic_tone(
            fundamental=440, duration=1.0, n_harmonics=1
        )
        audio3, _ = generate_harmonic_tone(
            fundamental=440, duration=1.0, n_harmonics=5
        )

        # More harmonics = more complex waveform
        # Check by comparing zero crossings or spectral complexity
        # Simple check: signals should be different
        assert not np.allclose(audio1, audio3)


class TestGenerateClick:
    """Tests for generate_click function."""

    def test_basic_click(self):
        """Test basic click generation."""
        from fiftyone.utils.audio import generate_click

        audio, sr = generate_click(
            duration=1.0,
            click_time=0.5,
            sample_rate=44100,
        )

        assert isinstance(audio, np.ndarray)
        assert len(audio) == 44100

    def test_click_location(self):
        """Test click appears at correct location."""
        from fiftyone.utils.audio import generate_click

        audio, sr = generate_click(
            duration=1.0,
            click_time=0.5,
        )

        # Find max amplitude location
        max_idx = np.argmax(np.abs(audio))
        max_time = max_idx / sr

        # Should be near 0.5s
        assert abs(max_time - 0.5) < 0.01


class TestGenerateTestAudioSuite:
    """Tests for generate_test_audio_suite function."""

    def test_generate_suite(self, temp_dir):
        """Test generating full test suite."""
        from fiftyone.utils.audio import generate_test_audio_suite

        files = generate_test_audio_suite(temp_dir)

        assert isinstance(files, list)
        assert len(files) > 0

        for f in files:
            assert os.path.exists(f)
            assert f.endswith(".wav")

    def test_suite_variety(self, temp_dir):
        """Test suite contains variety of signals."""
        from fiftyone.utils.audio import generate_test_audio_suite

        files = generate_test_audio_suite(temp_dir)

        # Should have multiple types
        names = [os.path.basename(f) for f in files]
        assert any("sine" in n for n in names)
        assert any("chirp" in n for n in names)
        assert any("noise" in n or "white" in n or "pink" in n for n in names)

    def test_custom_duration(self, temp_dir):
        """Test custom duration parameter."""
        from fiftyone.utils.audio import generate_test_audio_suite
        from fiftyone.utils.audio import load_audio

        files = generate_test_audio_suite(temp_dir, duration=2.0)

        # Check duration of generated files
        audio, sr = load_audio(files[0])
        duration = len(audio) / sr

        # Should be approximately 2 seconds
        assert abs(duration - 2.0) < 0.1

    def test_custom_sample_rate(self, temp_dir):
        """Test custom sample rate parameter."""
        from fiftyone.utils.audio import generate_test_audio_suite
        from fiftyone.utils.audio import load_audio

        files = generate_test_audio_suite(temp_dir, sample_rate=22050)

        audio, sr = load_audio(files[0])
        assert sr == 22050
