"""
Example: Audio Similarity Analysis with FiftyOne

This example demonstrates how to:
1. Create a dataset from audio files
2. Generate spectrograms for visualization
3. Compute FFT-based features for similarity comparison
4. Display similarity scores in the FiftyOne UI

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import tempfile

import fiftyone as fo


def run_audio_similarity_example():
    """Run the audio similarity example."""
    # Import audio utilities
    from fiftyone.utils.audio import (
        generate_test_audio_suite,
        create_audio_dataset,
        compute_dataset_fft_features,
        compute_dataset_similarity,
        SpectrogramConfig,
    )

    # Create temporary directory for test audio
    with tempfile.TemporaryDirectory() as tmpdir:
        print("=" * 60)
        print("FiftyOne Audio Similarity Example")
        print("=" * 60)

        # Step 1: Generate test audio files
        print("\n1. Generating test audio files...")
        audio_dir = os.path.join(tmpdir, "audio")
        audio_files = generate_test_audio_suite(audio_dir)
        print(f"   Generated {len(audio_files)} audio files")

        # Step 2: Create spectrogram configuration
        print("\n2. Configuring spectrogram parameters...")
        spectrogram_config = SpectrogramConfig(
            n_fft=2048,
            hop_length=512,
            colormap="viridis",
            window="hann",
            db_min=-80,
            db_max=0,
        )
        print(f"   FFT size: {spectrogram_config.n_fft}")
        print(f"   Hop length: {spectrogram_config.hop_length}")
        print(f"   Overlap: {spectrogram_config.overlap_ratio * 100:.1f}%")
        print(f"   Colormap: {spectrogram_config.colormap}")

        # Step 3: Create dataset with spectrograms
        print("\n3. Creating FiftyOne dataset with spectrograms...")
        dataset = create_audio_dataset(
            name="audio-similarity-example",
            audio_paths=audio_files,
            spectrogram_config=spectrogram_config,
            generate_spectrograms=True,
            compute_metadata=True,
            persistent=False,
        )
        print(f"   Created dataset with {len(dataset)} samples")

        # Step 4: Compute FFT features for each audio file
        print("\n4. Computing FFT features for similarity analysis...")
        compute_dataset_fft_features(
            dataset,
            audio_path_field="audio_path",
            output_field="fft_features",
            n_fft=2048,
            n_features=128,
            aggregation="mean",
            progress=True,
        )
        print("   FFT features computed")

        # Step 5: Compute similarity against the mean spectrum
        print("\n5. Computing similarity scores against mean spectrum...")
        compute_dataset_similarity(
            dataset,
            features_field="fft_features",
            output_field="similarity_to_mean",
            method="cosine",
            progress=True,
        )
        print("   Similarity scores computed")

        # Step 6: Find the reference sample (first sine wave)
        print("\n6. Computing similarity against a reference sample...")
        # Find the 440Hz sine wave as reference
        reference_sample = None
        for sample in dataset:
            if "sine_440hz" in sample.audio_path:
                reference_sample = sample
                break

        if reference_sample:
            compute_dataset_similarity(
                dataset,
                reference_sample_id=reference_sample.id,
                features_field="fft_features",
                output_field="similarity_to_440hz",
                method="cosine",
                progress=True,
            )
            print(
                f"   Reference: {os.path.basename(reference_sample.audio_path)}"
            )

        # Step 7: Display results
        print("\n7. Results:")
        print("-" * 60)
        print(f"{'Audio File':<40} {'Sim to Mean':>12} {'Sim to 440Hz':>12}")
        print("-" * 60)

        for sample in dataset.sort_by("similarity_to_mean", reverse=True):
            filename = os.path.basename(sample.audio_path)
            sim_mean = sample.get("similarity_to_mean", 0)
            sim_440 = sample.get("similarity_to_440hz", 0)
            print(f"{filename:<40} {sim_mean:>12.4f} {sim_440:>12.4f}")

        # Step 8: Show statistics
        print("\n8. Similarity Statistics:")
        sim_values = [
            s.similarity_to_mean
            for s in dataset
            if s.similarity_to_mean is not None
        ]
        if sim_values:
            print(
                f"   Mean similarity: {sum(sim_values) / len(sim_values):.4f}"
            )
            print(f"   Max similarity:  {max(sim_values):.4f}")
            print(f"   Min similarity:  {min(sim_values):.4f}")

        # Step 9: Launch FiftyOne App
        print("\n9. Launching FiftyOne App...")
        print("   The spectrograms will be displayed as images in the UI")
        print("   Similarity scores are shown in the sample fields")
        print(
            "   Sort by 'similarity_to_mean' or 'similarity_to_440hz' to explore"
        )

        # Create a view sorted by similarity
        view = dataset.sort_by("similarity_to_mean", reverse=True)

        session = fo.launch_app(view)

        print("\n" + "=" * 60)
        print("Press Ctrl+C to exit")
        print("=" * 60)

        # Keep the session alive
        session.wait()


if __name__ == "__main__":
    run_audio_similarity_example()
