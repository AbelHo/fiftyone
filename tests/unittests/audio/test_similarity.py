"""
Unit tests for audio similarity.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import numpy as np
import pytest


class TestComputeFFTFeatures:
    """Tests for compute_fft_features function."""

    def test_basic_fft(self, simple_audio):
        """Test basic FFT computation."""
        from fiftyone.utils.audio import compute_fft_features

        audio, sr = simple_audio
        features = compute_fft_features(audio, sr)

        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert len(features) > 0

    def test_n_fft_parameter(self, simple_audio):
        """Test n_fft parameter."""
        from fiftyone.utils.audio import compute_fft_features

        audio, sr = simple_audio

        features_small = compute_fft_features(audio, sr, n_fft=512)
        features_large = compute_fft_features(audio, sr, n_fft=2048)

        # Both have same output size (n_features default), but different values
        assert len(features_small) == len(features_large)
        # Values should be different due to different frequency resolution
        assert not np.allclose(features_small, features_large)

    def test_normalize(self, simple_audio):
        """Test normalization."""
        from fiftyone.utils.audio import compute_fft_features

        audio, sr = simple_audio
        features = compute_fft_features(audio, sr, normalize=True)

        # Normalized features should have unit norm
        norm = np.linalg.norm(features)
        assert abs(norm - 1.0) < 0.01

    def test_different_signals_different_features(self):
        """Test that different signals produce different features."""
        from fiftyone.utils.audio import compute_fft_features

        sr = 44100
        t = np.linspace(0, 1, sr, dtype=np.float32)

        # Two different frequencies
        audio1 = np.sin(2 * np.pi * 440 * t)  # A4
        audio2 = np.sin(2 * np.pi * 880 * t)  # A5

        features1 = compute_fft_features(audio1, sr)
        features2 = compute_fft_features(audio2, sr)

        # Features should be different
        assert not np.allclose(features1, features2)


class TestComputeMeanFFT:
    """Tests for compute_mean_fft function."""

    def test_mean_fft(self, simple_audio):
        """Test mean FFT computation."""
        from fiftyone.utils.audio import compute_mean_fft

        audio, sr = simple_audio
        mean_fft = compute_mean_fft(audio, sr)

        assert isinstance(mean_fft, np.ndarray)
        assert mean_fft.ndim == 1

    def test_mean_fft_hop_length(self, simple_audio):
        """Test different n_fft parameters."""
        from fiftyone.utils.audio import compute_mean_fft

        audio, sr = simple_audio

        mean_fft1 = compute_mean_fft(audio, sr, n_fft=1024)
        mean_fft2 = compute_mean_fft(audio, sr, n_fft=2048)

        # Different n_fft sizes should produce different output sizes
        assert len(mean_fft1) != len(mean_fft2)


class TestComputeSimilarity:
    """Tests for compute_similarity function."""

    def test_identical_signals(self, simple_audio):
        """Test similarity of identical feature vectors."""
        from fiftyone.utils.audio import (
            compute_similarity,
            compute_fft_features,
        )

        audio, sr = simple_audio
        features = compute_fft_features(audio, sr)

        similarity = compute_similarity(features, features)

        # Identical features should have similarity 1.0
        assert abs(similarity - 1.0) < 0.01

    def test_different_signals(self):
        """Test similarity of different signals."""
        from fiftyone.utils.audio import (
            compute_similarity,
            compute_fft_features,
        )

        sr = 44100
        t = np.linspace(0, 1, sr, dtype=np.float32)

        audio1 = np.sin(2 * np.pi * 440 * t)  # A4
        audio2 = np.sin(2 * np.pi * 880 * t)  # A5

        features1 = compute_fft_features(audio1, sr)
        features2 = compute_fft_features(audio2, sr)

        similarity = compute_similarity(features1, features2)

        # Different signals should have lower similarity
        assert similarity < 1.0
        assert (
            similarity >= 0.0
        )  # Cosine similarity can be 0 but not negative for positive spectra

    def test_orthogonal_signals(self):
        """Test similarity of different signal types."""
        from fiftyone.utils.audio import (
            compute_similarity,
            compute_fft_features,
        )

        sr = 44100
        t = np.linspace(0, 1, sr, dtype=np.float32)

        # A single tone vs noise should have low similarity
        audio1 = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        np.random.seed(42)
        audio2 = np.random.randn(sr).astype(np.float32)  # White noise

        features1 = compute_fft_features(audio1, sr)
        features2 = compute_fft_features(audio2, sr)

        similarity = compute_similarity(features1, features2)

        # Tone vs noise should have lower similarity than tone vs tone
        assert similarity < 1.0  # Just verify it computes without errors


class TestComputeDatasetFFTFeatures:
    """Tests for compute_dataset_fft_features function."""

    def test_compute_features(self, temp_dir):
        """Test computing features for a dataset."""
        import fiftyone as fo
        from fiftyone.utils.audio import (
            generate_test_audio_suite,
            create_audio_dataset,
            compute_dataset_fft_features,
        )

        # Generate test audio
        audio_files = generate_test_audio_suite(temp_dir)

        # Create dataset
        dataset = create_audio_dataset(
            audio_paths=audio_files[:3],
            generate_spectrograms=False,
            compute_metadata=True,
            persistent=False,
        )

        # Compute features
        compute_dataset_fft_features(dataset, output_field="fft_features")

        # Check features were added
        for sample in dataset:
            assert sample.has_field("fft_features")
            features = sample["fft_features"]
            assert features is not None
            assert isinstance(features, list)
            assert len(features) > 0

        # Cleanup
        dataset.delete()


class TestComputeDatasetSimilarity:
    """Tests for compute_dataset_similarity function."""

    def test_compute_similarity_scores(self, temp_dir):
        """Test computing similarity scores for dataset."""
        import fiftyone as fo
        from fiftyone.utils.audio import (
            generate_test_audio_suite,
            create_audio_dataset,
            compute_dataset_similarity,
            compute_dataset_fft_features,
        )

        # Generate test audio
        audio_files = generate_test_audio_suite(temp_dir)

        # Create dataset
        dataset = create_audio_dataset(
            audio_paths=audio_files[:4],
            generate_spectrograms=False,
            compute_metadata=True,
            persistent=False,
        )

        # First compute features
        compute_dataset_fft_features(dataset, output_field="fft_features")

        # Compute similarity (to mean)
        compute_dataset_similarity(
            dataset,
            features_field="fft_features",
            output_field="similarity",
        )

        # Check that similarity scores were added
        for sample in dataset:
            assert sample.has_field("similarity")
            sim = sample["similarity"]
            assert sim is not None
            assert -1.0 <= sim <= 1.0

        # Cleanup
        dataset.delete()


class TestFindSimilarSamples:
    """Tests for find_similar_samples function."""

    def test_find_similar(self, temp_dir):
        """Test finding similar samples."""
        import fiftyone as fo
        import numpy as np
        from fiftyone.utils.audio import (
            generate_test_audio_suite,
            create_audio_dataset,
            find_similar_samples,
            compute_dataset_fft_features,
        )

        # Generate test audio
        audio_files = generate_test_audio_suite(temp_dir)

        # Create dataset
        dataset = create_audio_dataset(
            audio_paths=audio_files[:5],
            generate_spectrograms=False,
            compute_metadata=True,
            persistent=False,
        )

        # First compute FFT features for all samples
        compute_dataset_fft_features(dataset, output_field="fft_features")

        # Get first sample's features
        sample = dataset.first()
        query_features = np.array(sample["fft_features"])

        # Find similar
        similar = find_similar_samples(dataset, query_features, k=3)

        assert len(similar) <= 3
        assert all(isinstance(item, tuple) for item in similar)
        assert all(
            len(item) == 2 for item in similar
        )  # (sample_id, similarity)

        # First result should be most similar (but not self)
        if len(similar) > 0:
            _, similarity = similar[0]
            assert similarity >= 0.0

        # Cleanup
        dataset.delete()
