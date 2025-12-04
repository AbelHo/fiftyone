# ~ test audio generation
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
audio_files = generate_test_audio_suite("/tmp/fiftyone_test_audio_suite")

# Individual generators
sine = generate_sine_wave(frequency=440, duration=1.0, sample_rate=44100)
chirp = generate_chirp(freq_start=100, freq_end=1000, duration=2.0)
noise = generate_noise(noise_type="pink", duration=1.0)
burst = generate_tone_burst(frequency=1000, duration=0.5, onset_ratio=0.2)
harmonic = generate_harmonic_tone(fundamental=220, n_harmonics=5)
click = generate_click(position=0.5, duration=1.0)

# ~ create audio dataset, run detection, compute similarity features
import os
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
    audio_paths=[
        os.path.join("/tmp/fiftyone_test_audio_suite", f)
        for f in os.listdir("/tmp/fiftyone_test_audio_suite")
    ],
    name="my_audio_dataset0",
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
