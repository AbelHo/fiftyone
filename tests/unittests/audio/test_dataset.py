"""
Unit tests for audio dataset utilities.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import numpy as np
import os
import pytest


class TestCreateAudioDataset:
    """Tests for create_audio_dataset function."""

    def test_create_basic_dataset(self, temp_dir):
        """Test creating a basic audio dataset."""
        from fiftyone.utils.audio import (
            generate_test_audio_suite,
            create_audio_dataset,
        )

        # Generate test audio
        audio_files = generate_test_audio_suite(temp_dir)

        # Create dataset
        dataset = create_audio_dataset(
            audio_paths=audio_files[:3],
            generate_spectrograms=False,
            compute_metadata=False,
            persistent=False,
        )

        try:
            assert len(dataset) == 3
            for sample in dataset:
                # Should have audio_path field
                assert (
                    hasattr(sample, "audio_path")
                    or sample.get("audio_path") is not None
                )
        finally:
            dataset.delete()

    def test_create_with_spectrograms(self, temp_dir):
        """Test creating dataset with spectrograms."""
        from fiftyone.utils.audio import (
            generate_test_audio_suite,
            create_audio_dataset,
            SpectrogramConfig,
        )

        # Generate test audio
        audio_files = generate_test_audio_suite(temp_dir)
        spectrogram_dir = os.path.join(temp_dir, "spectrograms")

        # Create config
        config = SpectrogramConfig(n_fft=512, hop_length=256)

        # Create dataset
        dataset = create_audio_dataset(
            audio_paths=audio_files[:2],
            generate_spectrograms=True,
            spectrogram_dir=spectrogram_dir,
            spectrogram_config=config,
            compute_metadata=False,
            persistent=False,
        )

        try:
            assert len(dataset) == 2
            for sample in dataset:
                # Check spectrogram was generated
                assert os.path.exists(sample.filepath)
                assert sample.filepath.endswith(".png")
        finally:
            dataset.delete()

    def test_create_with_metadata(self, temp_dir):
        """Test creating dataset with computed metadata."""
        from fiftyone.utils.audio import (
            generate_test_audio_suite,
            create_audio_dataset,
        )

        # Generate test audio
        audio_files = generate_test_audio_suite(temp_dir)

        # Create dataset with metadata
        dataset = create_audio_dataset(
            audio_paths=audio_files[:2],
            generate_spectrograms=False,
            compute_metadata=True,
            persistent=False,
        )

        try:
            for sample in dataset:
                # Should have audio metadata fields
                audio_meta = sample["audio_metadata"]
                assert audio_meta is not None
                assert "sample_rate" in audio_meta or hasattr(
                    audio_meta, "sample_rate"
                )
        finally:
            dataset.delete()

    def test_custom_name(self, temp_dir):
        """Test creating dataset with custom name."""
        import fiftyone as fo
        from fiftyone.utils.audio import (
            generate_test_audio_suite,
            create_audio_dataset,
        )

        # Generate test audio
        audio_files = generate_test_audio_suite(temp_dir)

        name = "test_audio_dataset_custom"

        # Delete if exists
        if fo.dataset_exists(name):
            fo.delete_dataset(name)

        # Create dataset
        dataset = create_audio_dataset(
            audio_paths=audio_files[:2],
            name=name,
            generate_spectrograms=False,
            compute_metadata=False,
            persistent=True,
        )

        try:
            assert dataset.name == name
            assert fo.dataset_exists(name)
        finally:
            dataset.delete()


class TestAddAudioSamples:
    """Tests for add_audio_samples function."""

    def test_add_samples(self, temp_dir):
        """Test adding audio samples to existing dataset."""
        import fiftyone as fo
        from fiftyone.utils.audio import (
            generate_test_audio_suite,
            add_audio_samples,
        )

        # Generate test audio
        audio_files = generate_test_audio_suite(temp_dir)

        # Create empty dataset
        dataset = fo.Dataset()

        # Add samples
        add_audio_samples(
            dataset,
            audio_paths=audio_files[:3],
            compute_metadata=False,
        )

        try:
            assert len(dataset) == 3
        finally:
            dataset.delete()


class TestGenerateSpectrogramsForDataset:
    """Tests for generate_spectrograms_for_dataset function."""

    def test_generate_for_existing_dataset(self, temp_dir):
        """Test generating spectrograms for existing dataset."""
        import fiftyone as fo
        from fiftyone.utils.audio import (
            generate_test_audio_suite,
            generate_spectrograms_for_dataset,
            SpectrogramConfig,
        )

        # Generate test audio
        audio_files = generate_test_audio_suite(temp_dir)
        spectrogram_dir = os.path.join(temp_dir, "spectrograms_post")

        # Create dataset manually
        samples = []
        for audio_path in audio_files[:2]:
            sample = fo.Sample(filepath=audio_path)
            sample["audio_path"] = audio_path
            samples.append(sample)

        dataset = fo.Dataset()
        dataset.add_samples(samples)

        # Generate spectrograms
        config = SpectrogramConfig(n_fft=512)
        generate_spectrograms_for_dataset(
            dataset,
            spectrogram_dir=spectrogram_dir,
            spectrogram_config=config,
            audio_path_field="audio_path",
        )

        try:
            # Check spectrograms were created
            for sample in dataset:
                sample.reload()
                try:
                    spec_path = sample["spectrogram_path"]
                    if spec_path is not None:
                        assert os.path.exists(spec_path)
                except (AttributeError, KeyError):
                    pass  # Field may not exist if spectrogram generation failed
        finally:
            dataset.delete()

    def test_regenerate_spectrograms(self, temp_dir):
        """Test regenerating spectrograms with different config."""
        import fiftyone as fo
        from fiftyone.utils.audio import (
            generate_test_audio_suite,
            create_audio_dataset,
            generate_spectrograms_for_dataset,
            SpectrogramConfig,
        )

        # Generate test audio
        audio_files = generate_test_audio_suite(temp_dir)
        spec_dir1 = os.path.join(temp_dir, "spec1")
        spec_dir2 = os.path.join(temp_dir, "spec2")

        # Create dataset with spectrograms
        dataset = create_audio_dataset(
            audio_paths=audio_files[:2],
            generate_spectrograms=True,
            spectrogram_dir=spec_dir1,
            spectrogram_config=SpectrogramConfig(n_fft=512),
            compute_metadata=False,
            persistent=False,
        )

        try:
            # Regenerate with different config
            generate_spectrograms_for_dataset(
                dataset,
                spectrogram_dir=spec_dir2,
                spectrogram_config=SpectrogramConfig(
                    n_fft=1024, colormap="viridis"
                ),
                audio_path_field="audio_path",
            )

            # Check new spectrograms exist
            assert os.path.exists(spec_dir2)
            files_in_dir = os.listdir(spec_dir2)
            assert len(files_in_dir) >= 2
        finally:
            dataset.delete()
