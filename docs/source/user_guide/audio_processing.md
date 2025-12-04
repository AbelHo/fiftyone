# FiftyOne Audio Processing Module

A comprehensive audio processing module for FiftyOne that enables working with
audio datasets, generating spectrograms, running detections, and computing
audio similarity.

## Table of Contents

-   [Installation](#installation)
-   [Quick Start](#quick-start)
-   [Features](#features)
    -   [Spectrogram Generation](#spectrogram-generation)
    -   [Audio Dataset Management](#audio-dataset-management)
    -   [Detection API](#detection-api)
    -   [Audio Similarity](#audio-similarity)
    -   [Test Audio Generation](#test-audio-generation)
-   [API Reference](#api-reference)
-   [Examples](#examples)
-   [Operators (UI Integration)](#operators-ui-integration)

## Installation

The audio module is included with FiftyOne. It requires the following
dependencies:

```bash
pip install scipy soundfile numpy matplotlib
```

## Quick Start

```python
import fiftyone as fo
from fiftyone.utils.audio import (
    create_audio_dataset,
    generate_spectrograms_for_dataset,
    ThresholdDetector,
    run_detection,
    compute_dataset_fft_features,
)

# Create a dataset from audio files
dataset = create_audio_dataset(
    audio_paths=["audio1.wav", "audio2.wav", "audio3.wav"],
    name="my_audio_dataset",
    generate_spectrograms=True,
    compute_metadata=True,
)

# Run detection
detector = ThresholdDetector(threshold=0.1)
run_detection(dataset, detector, output_field="detections")

# Compute similarity features
compute_dataset_fft_features(dataset, output_field="fft_features")

# Launch the FiftyOne App
session = fo.launch_app(dataset)
```

## Features

### Spectrogram Generation

Generate configurable spectrograms from audio files with various visualization
options.

#### SpectrogramConfig

The `SpectrogramConfig` dataclass controls spectrogram generation:

```python
from fiftyone.utils.audio import SpectrogramConfig

config = SpectrogramConfig(
    n_fft=2048,  # FFT window size
    hop_length=512,  # Samples between frames
    window="hann",  # Window function
    colormap="viridis",  # Matplotlib colormap
    freq_min=0,  # Minimum frequency (Hz)
    freq_max=8000,  # Maximum frequency (Hz)
    db_min=-80,  # Minimum dB for display
    db_max=0,  # Maximum dB for display
    mel_scale=False,  # Use mel scale
    n_mels=128,  # Number of mel bands
    figsize=(10, 4),  # Figure size in inches
    dpi=100,  # Resolution
)
```

#### Available Options

**Colormaps:**

-   `viridis`, `plasma`, `inferno`, `magma`, `cividis`
-   `gray`, `hot`, `cool`, `jet`, `turbo`

**Window Functions:**

-   `hann`, `hamming`, `blackman`, `bartlett`, `kaiser`, `boxcar`

#### Generate Spectrograms

```python
from fiftyone.utils.audio import (
    generate_spectrogram,
    save_spectrogram,
    generate_spectrogram_image,
)

# Load audio
from fiftyone.utils.audio import load_audio

audio, sr = load_audio("my_audio.wav")

# Generate spectrogram data
spec_db, freqs, times = generate_spectrogram(audio, sr, config)

# Save spectrogram as image
save_spectrogram(audio, sr, "spectrogram.png", config)

# Get spectrogram as numpy array (RGB image)
image = generate_spectrogram_image(audio, sr, config)
```

### Audio Dataset Management

#### Creating Datasets

```python
from fiftyone.utils.audio import create_audio_dataset

# Create from list of paths
dataset = create_audio_dataset(
    audio_paths=["audio1.wav", "audio2.wav"],
    name="my_dataset",
    generate_spectrograms=True,  # Auto-generate spectrograms
    spectrogram_dir="spectrograms",  # Output directory
    compute_metadata=True,  # Extract audio metadata
    persistent=True,  # Save to database
)

# Create from directory
import glob

audio_files = glob.glob("/path/to/audio/*.wav")
dataset = create_audio_dataset(audio_paths=audio_files)
```

#### Adding Samples

```python
from fiftyone.utils.audio import add_audio_samples

# Add more samples to existing dataset
add_audio_samples(
    dataset,
    audio_paths=["new_audio1.wav", "new_audio2.wav"],
    compute_metadata=True,
)
```

#### Batch Spectrogram Generation

```python
from fiftyone.utils.audio import generate_spectrograms_for_dataset

# Generate spectrograms for all samples
generate_spectrograms_for_dataset(
    dataset,
    output_dir="spectrograms",
    output_field="spectrogram_path",
    config=SpectrogramConfig(colormap="plasma"),
    overwrite=False,  # Skip existing
)
```

### Detection API

Detect audio events and convert them to bounding boxes for spectrogram
visualization.

#### Threshold Detector

Detects regions where audio amplitude exceeds a threshold:

```python
from fiftyone.utils.audio import ThresholdDetector, run_detection

detector = ThresholdDetector(
    threshold=0.1,  # Amplitude threshold (0-1)
    min_duration=0.1,  # Minimum event duration (seconds)
    label="sound",  # Detection label
)

# Run on dataset
run_detection(dataset, detector, output_field="detections")

# Run on single file
results = detector.detect_file("audio.wav")
for result in results:
    print(f"{result.label}: {result.time_start:.2f}s - {result.time_end:.2f}s")
```

#### Spectral Peak Detector

Detects regions with spectral energy above a threshold in specific frequency
ranges:

```python
from fiftyone.utils.audio import SpectralPeakDetector

detector = SpectralPeakDetector(
    threshold_db=-40,  # Threshold in dB
    min_duration=0.05,  # Minimum duration
    freq_min=500,  # Minimum frequency (Hz)
    freq_max=4000,  # Maximum frequency (Hz)
)

run_detection(dataset, detector, output_field="spectral_events")
```

#### Custom Detectors

Create custom detectors with the factory function:

```python
from fiftyone.utils.audio import create_custom_detector, DetectionResult


def my_detector(audio, sr, **kwargs):
    # Your detection logic here
    results = []
    # Example: detect loud regions
    for start, end in find_loud_regions(audio, sr):
        results.append(
            DetectionResult(
                label="loud",
                confidence=0.9,
                time_start=start,
                time_end=end,
            )
        )
    return results


detector = create_custom_detector(my_detector, name="loud_detector")
run_detection(dataset, detector, output_field="loud_events")
```

#### Detection Bounding Boxes

Detections are automatically converted to bounding boxes for spectrogram
overlay:

```python
# Detection results include bounding box conversion
result = DetectionResult(
    label="event",
    time_start=1.0,
    time_end=2.0,
    freq_min=500,
    freq_max=2000,
)

# Convert to FiftyOne Detection with bounding box
detection = result.to_detection(
    duration=3.0,  # Audio duration
    sample_rate=44100,  # Sample rate
    freq_range=(0, 8000),  # Spectrogram frequency range
)
```

### Audio Similarity

Compute audio similarity using FFT-based features.

#### Feature Extraction

```python
from fiftyone.utils.audio import (
    compute_fft_features,
    compute_dataset_fft_features,
)

# Single audio
features = compute_fft_features(
    audio,
    sample_rate=44100,
    n_fft=2048,
    n_features=128,  # Number of frequency bins
    aggregation="mean",  # How to aggregate over time
    normalize=True,  # Normalize features
)

# Entire dataset
compute_dataset_fft_features(
    dataset,
    output_field="fft_features",
    n_fft=2048,
    n_features=128,
)
```

#### Similarity Computation

```python
from fiftyone.utils.audio import compute_similarity, compute_dataset_similarity

# Between two feature vectors
similarity = compute_similarity(features1, features2, method="cosine")

# Dataset-wide similarity to reference
compute_dataset_similarity(
    dataset,
    reference_sample_id="sample_id",  # Or provide reference_features
    features_field="fft_features",
    output_field="similarity",
    method="cosine",
)
```

#### Find Similar Samples

```python
from fiftyone.utils.audio import find_similar_samples

# Find k most similar samples to a query
similar = find_similar_samples(
    dataset,
    query_features=query_features,
    features_field="fft_features",
    method="cosine",
    k=10,
)

# Returns list of (sample_id, similarity_score) tuples
for sample_id, score in similar:
    print(f"{sample_id}: {score:.3f}")
```

### Test Audio Generation

Generate test audio files for development and testing:

```python
from fiftyone.utils.audio import (
    generate_test_audio_suite,
    generate_sine_wave,
    generate_chirp,
    generate_noise,
    generate_tone_burst,
    generate_harmonic_tone,
    generate_click,
)

# Generate complete test suite (22 files)
audio_files = generate_test_audio_suite("/output/dir")

# Individual generators
sine = generate_sine_wave(frequency=440, duration=1.0, sample_rate=44100)
chirp = generate_chirp(freq_start=100, freq_end=1000, duration=2.0)
noise = generate_noise(noise_type="pink", duration=1.0)
burst = generate_tone_burst(frequency=1000, duration=0.5, onset_ratio=0.2)
harmonic = generate_harmonic_tone(fundamental=220, n_harmonics=5)
click = generate_click(position=0.5, duration=1.0)
```

## API Reference

### Core Functions

| Function                                   | Description                                   |
| ------------------------------------------ | --------------------------------------------- |
| `load_audio(filepath)`                     | Load audio file, returns (audio, sample_rate) |
| `save_audio(filepath, audio, sample_rate)` | Save audio to file                            |
| `normalize_audio(audio)`                   | Normalize audio to [-1, 1]                    |

### Spectrogram Functions

| Function                                        | Description                    |
| ----------------------------------------------- | ------------------------------ |
| `generate_spectrogram(audio, sr, config)`       | Generate spectrogram array     |
| `save_spectrogram(audio, sr, path, config)`     | Save spectrogram image         |
| `generate_spectrogram_image(audio, sr, config)` | Get spectrogram as numpy image |

### Dataset Functions

| Function                                          | Description                     |
| ------------------------------------------------- | ------------------------------- |
| `create_audio_dataset(audio_paths, ...)`          | Create dataset from audio files |
| `add_audio_samples(dataset, audio_paths, ...)`    | Add samples to dataset          |
| `generate_spectrograms_for_dataset(dataset, ...)` | Batch generate spectrograms     |

### Detection Functions

| Function                                | Description                   |
| --------------------------------------- | ----------------------------- |
| `run_detection(dataset, detector, ...)` | Run detector on dataset       |
| `create_custom_detector(func)`          | Create detector from function |
| `ThresholdDetector`                     | Amplitude-based detector      |
| `SpectralPeakDetector`                  | Spectral energy detector      |

### Similarity Functions

| Function                                           | Description                     |
| -------------------------------------------------- | ------------------------------- |
| `compute_fft_features(audio, sr, ...)`             | Extract FFT features            |
| `compute_similarity(features1, features2, method)` | Compute similarity              |
| `compute_dataset_fft_features(dataset, ...)`       | Batch feature extraction        |
| `compute_dataset_similarity(dataset, ...)`         | Dataset similarity to reference |
| `find_similar_samples(dataset, query, k)`          | K-nearest similar samples       |

### Generator Functions

| Function                                          | Description                  |
| ------------------------------------------------- | ---------------------------- |
| `generate_test_audio_suite(output_dir)`           | Generate complete test suite |
| `generate_sine_wave(frequency, duration, sr)`     | Generate sine wave           |
| `generate_chirp(freq_start, freq_end, duration)`  | Generate frequency sweep     |
| `generate_noise(noise_type, duration)`            | Generate noise               |
| `generate_tone_burst(frequency, duration, onset)` | Generate tone burst          |

## Examples

### Example 1: Audio Classification with Spectrograms

```python
import fiftyone as fo
from fiftyone.utils.audio import (
    create_audio_dataset,
    SpectrogramConfig,
)

# Create dataset with mel spectrograms
config = SpectrogramConfig(
    mel_scale=True,
    n_mels=128,
    colormap="magma",
    freq_max=8000,
)

dataset = create_audio_dataset(
    audio_paths=audio_files,
    name="audio_classification",
    generate_spectrograms=True,
    spectrogram_config=config,
)

# Add classification labels
for sample in dataset:
    # Your classification logic
    sample["prediction"] = fo.Classification(label="speech")
    sample.save()

session = fo.launch_app(dataset)
```

### Example 2: Sound Event Detection

```python
import fiftyone as fo
from fiftyone.utils.audio import (
    create_audio_dataset,
    SpectralPeakDetector,
    run_detection,
)

dataset = create_audio_dataset(
    audio_paths=audio_files,
    generate_spectrograms=True,
)

# Detect spectral events
detector = SpectralPeakDetector(
    threshold_db=-30,
    freq_min=1000,
    freq_max=4000,
)

run_detection(
    dataset,
    detector,
    output_field="events",
    spectrogram_field="spectrogram_path",  # Store bounding boxes relative to spectrogram
)

# View in App - bounding boxes overlay on spectrograms
session = fo.launch_app(dataset)
```

### Example 3: Audio Similarity Search

```python
from fiftyone.utils.audio import (
    create_audio_dataset,
    compute_dataset_fft_features,
    find_similar_samples,
    load_audio,
    compute_fft_features,
)

# Create dataset and compute features
dataset = create_audio_dataset(audio_paths=audio_files)
compute_dataset_fft_features(dataset, output_field="features")

# Query with new audio
query_audio, sr = load_audio("query.wav")
query_features = compute_fft_features(query_audio, sr)

# Find similar
similar = find_similar_samples(
    dataset,
    query_features=query_features,
    features_field="features",
    k=5,
)

print("Most similar samples:")
for sample_id, score in similar:
    sample = dataset[sample_id]
    print(f"  {sample.filepath}: {score:.3f}")
```

## Operators (UI Integration)

The audio module includes FiftyOne operators for UI integration:

### AudioPlaybackOperator

Play audio files directly in the FiftyOne App using MediaPlayerView.

### GenerateSpectrogramsOperator

Generate spectrograms for selected samples from the UI.

### RunAudioDetectionOperator

Run audio detection on selected samples with configurable parameters.

### ComputeSimilarityOperator

Compute audio similarity from the UI with reference sample selection.

## Testing

Run the unit tests:

```bash
conda activate fiftyone
python -m pytest tests/unittests/audio/ -v
```

## License

Copyright 2017-2025, Voxel51, Inc.
