"""
Audio detection utilities.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple

import numpy as np

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.utils as fou

from .spectrogram import SpectrogramConfig, generate_spectrogram

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result from an audio detection.

    Attributes:
        label: the detection label/class
        confidence: detection confidence (0-1)
        time_start: start time in seconds
        time_end: end time in seconds
        freq_min: minimum frequency in Hz (optional)
        freq_max: maximum frequency in Hz (optional)
        attributes: additional attributes for the detection
    """

    label: str
    confidence: float
    time_start: float
    time_end: float
    freq_min: Optional[float] = None
    freq_max: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_bounding_box(
        self,
        duration: float,
        sample_rate: int,
        freq_range: Optional[Tuple[float, float]] = None,
    ) -> List[float]:
        """Convert detection to normalized bounding box coordinates.

        Args:
            duration: total audio duration in seconds
            sample_rate: audio sample rate
            freq_range: tuple of (min_freq, max_freq) for the spectrogram

        Returns:
            bounding box as [x, y, width, height] in normalized coordinates (0-1)
        """
        if freq_range is None:
            freq_range = (0, sample_rate / 2)

        # X coordinates (time)
        x = self.time_start / duration
        width = (self.time_end - self.time_start) / duration

        # Y coordinates (frequency) - inverted because image origin is top-left
        freq_min = (
            self.freq_min if self.freq_min is not None else freq_range[0]
        )
        freq_max = (
            self.freq_max if self.freq_max is not None else freq_range[1]
        )

        # Normalize to freq_range
        y_max = 1 - (freq_min - freq_range[0]) / (
            freq_range[1] - freq_range[0]
        )
        y_min = 1 - (freq_max - freq_range[0]) / (
            freq_range[1] - freq_range[0]
        )

        y = y_min
        height = y_max - y_min

        return [x, y, width, height]

    def to_detection(
        self,
        duration: float,
        sample_rate: int,
        freq_range: Optional[Tuple[float, float]] = None,
    ) -> fol.Detection:
        """Convert to a FiftyOne Detection object.

        Args:
            duration: total audio duration in seconds
            sample_rate: audio sample rate
            freq_range: tuple of (min_freq, max_freq) for the spectrogram

        Returns:
            a :class:`fiftyone.core.labels.Detection`
        """
        bbox = self.to_bounding_box(duration, sample_rate, freq_range)

        # Build attributes dict
        attrs = {}
        for key, value in self.attributes.items():
            if isinstance(value, bool):
                attrs[key] = fol.BooleanAttribute(value=value)
            elif isinstance(value, (int, float)):
                attrs[key] = fol.NumericAttribute(value=value)
            elif isinstance(value, str):
                attrs[key] = fol.CategoricalAttribute(value=value)
            else:
                attrs[key] = fol.Attribute(value=value)

        # Add time/frequency info as attributes
        attrs["time_start"] = fol.NumericAttribute(value=self.time_start)
        attrs["time_end"] = fol.NumericAttribute(value=self.time_end)
        if self.freq_min is not None:
            attrs["freq_min"] = fol.NumericAttribute(value=self.freq_min)
        if self.freq_max is not None:
            attrs["freq_max"] = fol.NumericAttribute(value=self.freq_max)

        return fol.Detection(
            label=self.label,
            bounding_box=bbox,
            confidence=self.confidence,
            attributes=attrs,
        )


class AudioDetector(ABC):
    """Abstract base class for audio detectors.

    Subclass this to create custom audio detection algorithms.
    """

    @abstractmethod
    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs,
    ) -> List[DetectionResult]:
        """Run detection on audio data.

        Args:
            audio: audio data as 1D numpy array
            sample_rate: sample rate in Hz
            **kwargs: additional arguments

        Returns:
            list of :class:`DetectionResult` objects
        """
        pass

    def detect_file(
        self,
        filepath: str,
        **kwargs,
    ) -> List[DetectionResult]:
        """Run detection on an audio file.

        Args:
            filepath: path to audio file
            **kwargs: additional arguments

        Returns:
            list of :class:`DetectionResult` objects
        """
        from .core import load_audio

        audio, sr = load_audio(filepath)
        return self.detect(audio, sr, **kwargs)


class ThresholdDetector(AudioDetector):
    """Simple threshold-based audio detector.

    Detects regions where the audio amplitude exceeds a threshold.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        min_duration: float = 0.1,
        label: str = "sound",
    ):
        """Initialize the detector.

        Args:
            threshold: amplitude threshold (0-1)
            min_duration: minimum detection duration in seconds
            label: label for detections
        """
        self.threshold = threshold
        self.min_duration = min_duration
        self.label = label

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs,
    ) -> List[DetectionResult]:
        """Detect regions above threshold."""
        # Compute envelope
        envelope = np.abs(audio)

        # Find regions above threshold
        above_threshold = envelope > self.threshold

        # Find transitions
        diff = np.diff(above_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        # Handle edge cases
        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(audio)]])

        # Convert to detections
        results = []
        min_samples = int(self.min_duration * sample_rate)

        for start, end in zip(starts, ends):
            if end - start >= min_samples:
                # Compute confidence as max amplitude in region
                confidence = float(np.max(envelope[start:end]))

                results.append(
                    DetectionResult(
                        label=self.label,
                        confidence=confidence,
                        time_start=start / sample_rate,
                        time_end=end / sample_rate,
                    )
                )

        return results


class SpectralPeakDetector(AudioDetector):
    """Detector that finds spectral peaks in the spectrogram."""

    def __init__(
        self,
        threshold_db: float = -40,
        min_duration: float = 0.05,
        freq_min: float = 100,
        freq_max: float = 8000,
        spectrogram_config: Optional[SpectrogramConfig] = None,
    ):
        """Initialize the detector.

        Args:
            threshold_db: threshold in dB for peak detection
            min_duration: minimum duration for detections
            freq_min: minimum frequency to consider
            freq_max: maximum frequency to consider
            spectrogram_config: spectrogram configuration
        """
        self.threshold_db = threshold_db
        self.min_duration = min_duration
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.spectrogram_config = spectrogram_config or SpectrogramConfig()

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs,
    ) -> List[DetectionResult]:
        """Detect spectral peaks."""
        from scipy import ndimage

        # Generate spectrogram
        config = SpectrogramConfig(
            **{
                **self.spectrogram_config.to_dict(),
                "freq_min": self.freq_min,
                "freq_max": self.freq_max,
                "normalize": False,
            }
        )
        spec_db, freqs, times = generate_spectrogram(
            audio, sample_rate, config
        )

        # Threshold - spec_db is in dB when normalize=False
        # Compare directly to threshold_db
        binary = spec_db > self.threshold_db

        # Label connected regions
        labeled, num_features = ndimage.label(binary)

        results = []
        for i in range(1, num_features + 1):
            # Get region
            region = labeled == i

            # Get bounding box
            rows = np.any(region, axis=1)
            cols = np.any(region, axis=0)

            if not np.any(rows) or not np.any(cols):
                continue

            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]

            freq_idx_min, freq_idx_max = row_indices[0], row_indices[-1]
            time_idx_min, time_idx_max = col_indices[0], col_indices[-1]

            # Convert to time/frequency
            time_start = times[time_idx_min]
            time_end = times[min(time_idx_max, len(times) - 1)]
            freq_start = freqs[freq_idx_min]
            freq_end = freqs[min(freq_idx_max, len(freqs) - 1)]

            # Check minimum duration
            if time_end - time_start < self.min_duration:
                continue

            # Compute confidence as mean intensity in region
            confidence = float(np.mean(spec_db[region]))

            results.append(
                DetectionResult(
                    label="spectral_peak",
                    confidence=confidence,
                    time_start=float(time_start),
                    time_end=float(time_end),
                    freq_min=float(freq_start),
                    freq_max=float(freq_end),
                )
            )

        return results


def run_detection(
    dataset: fo.Dataset,
    detector: AudioDetector,
    audio_path_field: str = "audio_path",
    output_field: str = "detections",
    freq_range: Optional[Tuple[float, float]] = None,
    progress: bool = True,
) -> None:
    """Run detection on all samples in a dataset.

    Args:
        dataset: the dataset to process
        detector: the detector to use
        audio_path_field: field containing audio file paths
        output_field: field to store detections
        freq_range: frequency range for bounding box conversion
        progress: whether to show progress bar
    """
    samples = list(dataset)
    with fou.ProgressBar(total=len(samples), progress=progress) as pb:
        for sample in pb(samples):
            # Access field directly using bracket notation
            if not sample.has_field(audio_path_field):
                continue
            audio_path = sample[audio_path_field]
            if audio_path is None:
                continue

            try:
                # Get audio metadata for duration
                duration = None
                sample_rate = None
                if sample.has_field("audio_metadata"):
                    audio_meta = sample["audio_metadata"]
                    if audio_meta is not None:
                        duration = getattr(audio_meta, "duration", None)
                        sample_rate = getattr(audio_meta, "sample_rate", None)

                if duration is None or sample_rate is None:
                    from .core import get_audio_info

                    info = get_audio_info(audio_path)
                    duration = info["duration"]
                    sample_rate = info["sample_rate"]

                # Run detection
                results = detector.detect_file(audio_path)

                # Convert to FiftyOne detections
                detections = []
                for result in results:
                    det = result.to_detection(
                        duration, sample_rate, freq_range
                    )
                    detections.append(det)

                sample[output_field] = fol.Detections(detections=detections)
                sample.save()

            except Exception as e:
                logger.warning(f"Detection failed for {audio_path}: {e}")


def create_custom_detector(
    detect_func: Callable[[np.ndarray, int], List[DetectionResult]],
) -> AudioDetector:
    """Create a custom detector from a function.

    Args:
        detect_func: function that takes (audio, sample_rate) and returns
            list of DetectionResult

    Returns:
        an :class:`AudioDetector` instance

    Example::

        def my_detect(audio, sr):
            # Your detection logic here
            return [DetectionResult(label="found", confidence=0.9,
                                   time_start=0.5, time_end=1.0)]

        detector = create_custom_detector(my_detect)
    """

    class CustomDetector(AudioDetector):
        def __init__(self, func):
            self._func = func

        def detect(self, audio, sample_rate, **kwargs):
            return self._func(audio, sample_rate)

    return CustomDetector(detect_func)
