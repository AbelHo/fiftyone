"""
Example: Audio Classifier with Bounding Boxes

This example demonstrates how to:
1. Create a dataset from audio files
2. Generate spectrograms for visualization
3. Run a simple classifier on the spectrograms
4. Add bounding box detections for audio events
5. Display results in the FiftyOne UI with bounding boxes

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import tempfile
from typing import List, Tuple

import numpy as np

import fiftyone as fo
import fiftyone.core.labels as fol


def simple_spectrogram_classifier(
    spectrogram: np.ndarray,
) -> Tuple[str, float]:
    """A simple classifier based on spectrogram statistics.

    This is a demonstration classifier that classifies audio based on
    simple spectral characteristics.

    Args:
        spectrogram: 2D spectrogram array

    Returns:
        tuple of (class_label, confidence)
    """
    # Compute simple features
    mean_energy = np.mean(spectrogram)
    std_energy = np.std(spectrogram)
    max_energy = np.max(spectrogram)

    # Spectral centroid approximation
    freq_bins = np.arange(spectrogram.shape[0])
    spectral_centroid = np.sum(freq_bins[:, None] * spectrogram) / (
        np.sum(spectrogram) + 1e-10
    )
    spectral_centroid_normalized = spectral_centroid / spectrogram.shape[0]

    # Simple rule-based classification
    if std_energy > 0.3:
        # High variation - likely noise or complex signal
        if spectral_centroid_normalized > 0.5:
            return "high_frequency_noise", 0.7 + 0.3 * std_energy
        else:
            return "low_frequency_noise", 0.7 + 0.3 * std_energy
    elif max_energy > 0.8 and std_energy < 0.2:
        # Strong, consistent signal - likely a tone
        if spectral_centroid_normalized > 0.6:
            return "high_tone", 0.8 + 0.2 * max_energy
        elif spectral_centroid_normalized > 0.3:
            return "mid_tone", 0.8 + 0.2 * max_energy
        else:
            return "low_tone", 0.8 + 0.2 * max_energy
    elif mean_energy < 0.2:
        return "quiet", 0.9 - mean_energy
    else:
        # Check for chirp-like patterns (increasing/decreasing centroid over time)
        left_half = spectrogram[:, : spectrogram.shape[1] // 2]
        right_half = spectrogram[:, spectrogram.shape[1] // 2 :]

        left_centroid = np.sum(freq_bins[:, None] * left_half) / (
            np.sum(left_half) + 1e-10
        )
        right_centroid = np.sum(freq_bins[:, None] * right_half) / (
            np.sum(right_half) + 1e-10
        )

        if right_centroid > left_centroid * 1.3:
            return "ascending_chirp", 0.75
        elif left_centroid > right_centroid * 1.3:
            return "descending_chirp", 0.75
        else:
            return "complex_signal", 0.6


def detect_events_in_spectrogram(
    spectrogram: np.ndarray,
    threshold: float = 0.3,
    min_duration_frames: int = 5,
) -> List[fol.Detection]:
    """Detect events in a spectrogram and return bounding boxes.

    Args:
        spectrogram: 2D spectrogram array (frequency x time)
        threshold: energy threshold for detection
        min_duration_frames: minimum event duration in frames

    Returns:
        list of Detection objects with bounding boxes
    """
    from scipy import ndimage

    # Threshold the spectrogram
    binary = spectrogram > threshold

    # Label connected components
    labeled, num_features = ndimage.label(binary)

    detections = []

    for i in range(1, num_features + 1):
        region = labeled == i

        # Get bounding box
        rows = np.any(region, axis=1)
        cols = np.any(region, axis=0)

        if not np.any(rows) or not np.any(cols):
            continue

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        y_min = row_indices[0] / spectrogram.shape[0]
        y_max = row_indices[-1] / spectrogram.shape[0]
        x_min = col_indices[0] / spectrogram.shape[1]
        x_max = col_indices[-1] / spectrogram.shape[1]

        # Check minimum duration
        if (x_max - x_min) * spectrogram.shape[1] < min_duration_frames:
            continue

        # Compute confidence based on mean energy in region
        confidence = float(np.mean(spectrogram[region]))

        # Classify the region
        region_spec = spectrogram[
            row_indices[0] : row_indices[-1] + 1,
            col_indices[0] : col_indices[-1] + 1,
        ]
        label, _ = simple_spectrogram_classifier(region_spec)

        # Convert to FiftyOne format (y is inverted for image coordinates)
        # Format: [x, y, width, height] all in [0, 1]
        bbox = [
            x_min,
            1 - y_max,  # Invert y because spectrogram origin is bottom-left
            x_max - x_min,
            y_max - y_min,
        ]

        detection = fol.Detection(
            label=label,
            bounding_box=bbox,
            confidence=confidence,
        )
        detections.append(detection)

    return detections


def run_audio_classifier_example():
    """Run the audio classifier with bounding boxes example."""
    from fiftyone.utils.audio import (
        generate_test_audio_suite,
        create_audio_dataset,
        load_audio,
        generate_spectrogram,
        SpectrogramConfig,
    )

    # Create temporary directory for test audio
    with tempfile.TemporaryDirectory() as tmpdir:
        print("=" * 60)
        print("FiftyOne Audio Classifier with Bounding Boxes Example")
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
            normalize=True,
        )

        # Step 3: Create dataset with spectrograms
        print("\n3. Creating FiftyOne dataset with spectrograms...")
        dataset = create_audio_dataset(
            name="audio-classifier-example",
            audio_paths=audio_files,
            spectrogram_config=spectrogram_config,
            generate_spectrograms=True,
            compute_metadata=True,
            persistent=False,
        )
        print(f"   Created dataset with {len(dataset)} samples")

        # Step 4: Run classification and event detection on each sample
        print("\n4. Running classification and event detection...")

        total_detections = 0
        for sample in dataset:
            audio_path = sample.audio_path

            try:
                # Load audio and generate spectrogram
                audio, sr = load_audio(audio_path)
                spec, freqs, times = generate_spectrogram(
                    audio, sr, spectrogram_config
                )

                # Classify the entire spectrogram
                overall_class, overall_conf = simple_spectrogram_classifier(
                    spec
                )
                sample["classification"] = fol.Classification(
                    label=overall_class,
                    confidence=overall_conf,
                )

                # Detect events and create bounding boxes
                detections = detect_events_in_spectrogram(
                    spec,
                    threshold=0.3,
                    min_duration_frames=5,
                )

                if detections:
                    sample["detections"] = fol.Detections(
                        detections=detections
                    )
                    total_detections += len(detections)

                sample.save()

            except Exception as e:
                print(
                    f"   Warning: Failed to process {os.path.basename(audio_path)}: {e}"
                )

        print(f"   Processed {len(dataset)} samples")
        print(f"   Total detections: {total_detections}")

        # Step 5: Display results
        print("\n5. Classification Results:")
        print("-" * 60)
        print(f"{'Audio File':<35} {'Class':<20} {'Confidence':>10}")
        print("-" * 60)

        for sample in dataset:
            filename = os.path.basename(sample.audio_path)
            classification = sample.get("classification")
            if classification:
                print(
                    f"{filename:<35} {classification.label:<20} {classification.confidence:>10.4f}"
                )

        # Step 6: Show detection statistics
        print("\n6. Detection Statistics:")
        print("-" * 60)

        class_counts = {}
        for sample in dataset:
            dets = sample.get("detections")
            if dets:
                for det in dets.detections:
                    class_counts[det.label] = (
                        class_counts.get(det.label, 0) + 1
                    )

        print(f"{'Detection Class':<25} {'Count':>10}")
        print("-" * 60)
        for label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"{label:<25} {count:>10}")

        # Step 7: Launch FiftyOne App
        print("\n7. Launching FiftyOne App...")
        print("   The spectrograms will be displayed as images")
        print("   Bounding boxes show detected audio events")
        print("   Classifications are shown in sample fields")
        print(
            "   Filter by 'classification.label' to explore different classes"
        )

        session = fo.launch_app(dataset)

        print("\n" + "=" * 60)
        print("Tips:")
        print("  - Use the filter panel to filter by classification label")
        print("  - Click on samples to see detailed bounding boxes")
        print("  - The bounding boxes represent time-frequency regions")
        print("  - Sort by classification confidence to see top results")
        print("=" * 60)
        print("\nPress Ctrl+C to exit")

        session.wait()


if __name__ == "__main__":
    run_audio_classifier_example()
