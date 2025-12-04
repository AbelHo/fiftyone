"""
Unit tests for audio detection.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import numpy as np
import pytest


class TestDetectionResult:
    """Tests for DetectionResult class."""

    def test_basic_detection(self):
        """Test basic detection result."""
        from fiftyone.utils.audio import DetectionResult

        result = DetectionResult(
            label="test",
            confidence=0.9,
            time_start=0.5,
            time_end=1.5,
        )

        assert result.label == "test"
        assert result.confidence == 0.9
        assert result.time_start == 0.5
        assert result.time_end == 1.5

    def test_to_bounding_box(self):
        """Test conversion to bounding box."""
        from fiftyone.utils.audio import DetectionResult

        result = DetectionResult(
            label="test",
            confidence=0.9,
            time_start=0.0,
            time_end=1.0,
            freq_min=0,
            freq_max=22050,
        )

        bbox = result.to_bounding_box(
            duration=2.0,
            sample_rate=44100,
        )

        # bbox format: [x, y, width, height]
        assert len(bbox) == 4
        assert bbox[0] == 0.0  # x start
        assert bbox[2] == 0.5  # width (1s / 2s)

    def test_to_detection(self):
        """Test conversion to FiftyOne Detection."""
        from fiftyone.utils.audio import DetectionResult
        import fiftyone.core.labels as fol

        result = DetectionResult(
            label="sound",
            confidence=0.85,
            time_start=0.5,
            time_end=1.0,
        )

        detection = result.to_detection(duration=2.0, sample_rate=44100)

        assert isinstance(detection, fol.Detection)
        assert detection.label == "sound"
        assert detection.confidence == 0.85


class TestThresholdDetector:
    """Tests for ThresholdDetector class."""

    def test_detect_simple(self):
        """Test detection on simple signal."""
        from fiftyone.utils.audio.detection import ThresholdDetector

        # Create signal with loud section in the middle
        sr = 44100
        duration = 2.0
        audio = np.zeros(int(sr * duration), dtype=np.float32)

        # Add loud section from 0.5s to 1.0s
        start_sample = int(0.5 * sr)
        end_sample = int(1.0 * sr)
        audio[start_sample:end_sample] = 0.5

        detector = ThresholdDetector(
            threshold=0.1, min_duration=0.1, label="event"
        )
        results = detector.detect(audio, sr)

        assert len(results) >= 1
        # Should detect the loud section
        found_event = any(
            0.4 <= r.time_start <= 0.6 and 0.9 <= r.time_end <= 1.1
            for r in results
        )
        assert found_event

    def test_detect_no_events(self):
        """Test detection on quiet signal."""
        from fiftyone.utils.audio.detection import ThresholdDetector

        sr = 44100
        audio = np.zeros(sr, dtype=np.float32)  # Silent

        detector = ThresholdDetector(threshold=0.1)
        results = detector.detect(audio, sr)

        assert len(results) == 0

    def test_detect_multiple_events(self):
        """Test detection of multiple events."""
        from fiftyone.utils.audio.detection import ThresholdDetector

        sr = 44100
        duration = 3.0
        audio = np.zeros(int(sr * duration), dtype=np.float32)

        # Add two events
        audio[int(0.5 * sr) : int(0.8 * sr)] = 0.5  # Event 1
        audio[int(1.5 * sr) : int(2.0 * sr)] = 0.5  # Event 2

        detector = ThresholdDetector(threshold=0.1, min_duration=0.1)
        results = detector.detect(audio, sr)

        assert len(results) == 2


class TestSpectralPeakDetector:
    """Tests for SpectralPeakDetector class."""

    def test_detect_tone(self):
        """Test detection of a tone."""
        from fiftyone.utils.audio.detection import SpectralPeakDetector

        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1kHz tone

        detector = SpectralPeakDetector(
            threshold_db=-50,
            freq_min=500,
            freq_max=2000,
        )
        results = detector.detect(audio, sr)

        # Should detect at least one spectral peak
        assert len(results) >= 1

    def test_detect_with_frequency_bounds(self):
        """Test frequency-bounded detection."""
        from fiftyone.utils.audio.detection import SpectralPeakDetector

        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 5000 * t)  # 5kHz tone

        # Detector looking at low frequencies should not find this
        detector = SpectralPeakDetector(
            threshold_db=-50,
            freq_min=100,
            freq_max=1000,
        )
        results = detector.detect(audio, sr)

        # Most detections should be filtered out
        high_freq_detections = [
            r for r in results if r.freq_min is not None and r.freq_min > 1000
        ]
        # With strict frequency bounds, shouldn't detect 5kHz tone
        assert len(high_freq_detections) == 0


class TestRunDetection:
    """Tests for run_detection function."""

    def test_run_on_dataset(self, temp_dir):
        """Test running detection on a dataset."""
        import fiftyone as fo
        from fiftyone.utils.audio import (
            generate_test_audio_suite,
            create_audio_dataset,
            ThresholdDetector,
            run_detection,
        )

        # Generate test audio
        audio_files = generate_test_audio_suite(temp_dir)

        # Filter for audio types suitable for threshold detection
        # (tone bursts, clicks, multi-event have clear amplitude regions)
        suitable_files = [
            f
            for f in audio_files
            if any(
                pattern in f
                for pattern in ["tone_burst", "click", "multi_event"]
            )
        ]

        # Fallback to first 5 if no suitable files found
        if not suitable_files:
            suitable_files = audio_files[:5]

        # Create dataset
        dataset = create_audio_dataset(
            audio_paths=suitable_files[:5],  # Use subset for speed
            generate_spectrograms=False,
            compute_metadata=True,
            persistent=False,
        )

        # Run detection with low threshold and very short min_duration
        # to detect transient sounds
        detector = ThresholdDetector(threshold=0.05, min_duration=0.01)
        run_detection(dataset, detector, output_field="audio_detections")

        # Check that detection field was added to all samples
        for sample in dataset:
            assert sample.has_field("audio_detections")
            dets = sample["audio_detections"]
            # Detections object should exist (may be empty for some samples)
            assert dets is not None

        # Cleanup
        dataset.delete()


class TestCustomDetector:
    """Tests for create_custom_detector function."""

    def test_custom_detector(self):
        """Test creating a custom detector."""
        from fiftyone.utils.audio import (
            create_custom_detector,
            DetectionResult,
        )

        def my_detect(audio, sr):
            # Simple detector that always finds one event
            return [
                DetectionResult(
                    label="custom",
                    confidence=0.95,
                    time_start=0.0,
                    time_end=len(audio) / sr,
                )
            ]

        detector = create_custom_detector(my_detect)

        audio = np.zeros(44100, dtype=np.float32)
        results = detector.detect(audio, 44100)

        assert len(results) == 1
        assert results[0].label == "custom"
        assert results[0].confidence == 0.95
