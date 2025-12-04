"""
Test audio file generator utilities.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def generate_sine_wave(
    frequency: float,
    duration: float,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    phase: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """Generate a sine wave.

    Args:
        frequency: frequency in Hz
        duration: duration in seconds
        sample_rate: sample rate in Hz
        amplitude: amplitude (0-1)
        phase: initial phase in radians

    Returns:
        tuple of (audio_data, sample_rate)
    """
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    audio = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return audio.astype(np.float32), sample_rate


def generate_chirp(
    freq_start: float,
    freq_end: float,
    duration: float,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    method: str = "linear",
) -> Tuple[np.ndarray, int]:
    """Generate a chirp (frequency sweep) signal.

    Args:
        freq_start: starting frequency in Hz
        freq_end: ending frequency in Hz
        duration: duration in seconds
        sample_rate: sample rate in Hz
        amplitude: amplitude (0-1)
        method: sweep method, one of "linear", "quadratic", "logarithmic"

    Returns:
        tuple of (audio_data, sample_rate)
    """
    from scipy import signal

    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    audio = amplitude * signal.chirp(
        t, freq_start, duration, freq_end, method=method
    )
    return audio.astype(np.float32), sample_rate


def generate_noise(
    duration: float,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    noise_type: str = "white",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Generate noise signal.

    Args:
        duration: duration in seconds
        sample_rate: sample rate in Hz
        amplitude: amplitude (0-1)
        noise_type: type of noise, one of:
            - "white": white noise (flat spectrum)
            - "pink": pink noise (1/f spectrum)
            - "brown": brown/red noise (1/f^2 spectrum)
        seed: random seed for reproducibility

    Returns:
        tuple of (audio_data, sample_rate)
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = int(duration * sample_rate)

    if noise_type == "white":
        audio = np.random.randn(n_samples)

    elif noise_type == "pink":
        # Pink noise using Voss-McCartney algorithm
        audio = _generate_pink_noise(n_samples)

    elif noise_type == "brown":
        # Brown noise is integrated white noise
        white = np.random.randn(n_samples)
        audio = np.cumsum(white)
        # Normalize to prevent overflow
        audio = audio / np.max(np.abs(audio))

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Normalize and scale
    audio = audio / np.max(np.abs(audio))
    audio = amplitude * audio

    return audio.astype(np.float32), sample_rate


def _generate_pink_noise(n_samples: int, num_rows: int = 16) -> np.ndarray:
    """Generate pink noise using Voss-McCartney algorithm."""
    array = np.empty((n_samples, num_rows))
    array.fill(np.nan)
    array[0, :] = np.random.random(num_rows)
    array[:, 0] = np.random.random(n_samples)

    # Number of samples that can be generated
    cols = np.random.geometric(0.5, n_samples)
    cols[cols >= num_rows] = 0
    rows = np.random.random(n_samples)

    array[np.arange(n_samples), cols] = rows
    array = np.nanmean(array, axis=1)

    # Fill remaining NaN values
    mask = np.isnan(array)
    array[mask] = np.interp(
        np.flatnonzero(mask), np.flatnonzero(~mask), array[~mask]
    )

    return array - np.mean(array)


def generate_tone_burst(
    frequency: float,
    burst_duration: float,
    total_duration: float,
    onset_time: float,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    envelope: str = "rectangular",
    attack: float = 0.01,
    release: float = 0.01,
) -> Tuple[np.ndarray, int]:
    """Generate a tone burst (tone with onset/offset).

    Args:
        frequency: tone frequency in Hz
        burst_duration: duration of the tone burst in seconds
        total_duration: total signal duration in seconds
        onset_time: when the burst starts in seconds
        sample_rate: sample rate in Hz
        amplitude: amplitude (0-1)
        envelope: envelope type, one of "rectangular", "hann", "adsr"
        attack: attack time for ADSR envelope
        release: release time for ADSR envelope

    Returns:
        tuple of (audio_data, sample_rate)
    """
    n_samples = int(total_duration * sample_rate)
    audio = np.zeros(n_samples, dtype=np.float32)

    # Generate tone
    burst_samples = int(burst_duration * sample_rate)
    t = np.arange(burst_samples) / sample_rate
    tone = amplitude * np.sin(2 * np.pi * frequency * t)

    # Apply envelope
    if envelope == "rectangular":
        env = np.ones(burst_samples)
    elif envelope == "hann":
        env = np.hanning(burst_samples)
    elif envelope == "adsr":
        env = _generate_adsr_envelope(
            burst_samples, sample_rate, attack, release
        )
    else:
        raise ValueError(f"Unknown envelope type: {envelope}")

    tone = tone * env

    # Place burst in signal
    onset_sample = int(onset_time * sample_rate)
    end_sample = min(onset_sample + burst_samples, n_samples)
    audio[onset_sample:end_sample] = tone[: end_sample - onset_sample]

    return audio.astype(np.float32), sample_rate


def _generate_adsr_envelope(
    n_samples: int,
    sample_rate: int,
    attack: float,
    release: float,
    sustain_level: float = 0.8,
) -> np.ndarray:
    """Generate ADSR envelope."""
    attack_samples = int(attack * sample_rate)
    release_samples = int(release * sample_rate)
    sustain_samples = n_samples - attack_samples - release_samples

    if sustain_samples < 0:
        # Not enough samples, scale attack and release
        total = attack_samples + release_samples
        attack_samples = int(attack_samples * n_samples / total)
        release_samples = n_samples - attack_samples
        sustain_samples = 0

    # Attack phase
    attack_env = np.linspace(0, 1, attack_samples)

    # Sustain phase
    sustain_env = np.ones(sustain_samples) * sustain_level

    # Release phase
    release_env = np.linspace(sustain_level, 0, release_samples)

    return np.concatenate([attack_env, sustain_env, release_env])


def generate_harmonic_tone(
    fundamental: float,
    duration: float,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    n_harmonics: int = 5,
    harmonic_decay: float = 0.5,
) -> Tuple[np.ndarray, int]:
    """Generate a tone with harmonics.

    Args:
        fundamental: fundamental frequency in Hz
        duration: duration in seconds
        sample_rate: sample rate in Hz
        amplitude: amplitude (0-1)
        n_harmonics: number of harmonics to include
        harmonic_decay: decay factor for harmonic amplitudes

    Returns:
        tuple of (audio_data, sample_rate)
    """
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    audio = np.zeros_like(t)

    for i in range(1, n_harmonics + 1):
        harmonic_amp = amplitude * (harmonic_decay ** (i - 1))
        audio += harmonic_amp * np.sin(2 * np.pi * fundamental * i * t)

    # Normalize
    audio = audio / np.max(np.abs(audio)) * amplitude

    return audio.astype(np.float32), sample_rate


def generate_click(
    duration: float,
    click_time: float,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
    click_duration: float = 0.001,
) -> Tuple[np.ndarray, int]:
    """Generate a click/impulse sound.

    Args:
        duration: total duration in seconds
        click_time: when the click occurs in seconds
        sample_rate: sample rate in Hz
        amplitude: amplitude (0-1)
        click_duration: duration of the click in seconds

    Returns:
        tuple of (audio_data, sample_rate)
    """
    n_samples = int(duration * sample_rate)
    audio = np.zeros(n_samples, dtype=np.float32)

    click_samples = int(click_duration * sample_rate)
    click_start = int(click_time * sample_rate)
    click_end = min(click_start + click_samples, n_samples)

    # Create click as a windowed impulse
    click = amplitude * np.hanning(click_samples)
    audio[click_start:click_end] = click[: click_end - click_start]

    return audio.astype(np.float32), sample_rate


def generate_test_audio_suite(
    output_dir: str,
    sample_rate: int = 44100,
    duration: float = 2.0,
    format: str = "wav",
) -> List[str]:
    """Generate a suite of test audio files.

    Creates various test signals for testing audio processing:
    - Sine waves at different frequencies
    - Chirps (frequency sweeps)
    - Noise (white, pink, brown)
    - Tone bursts
    - Harmonic tones
    - Clicks

    Args:
        output_dir: directory to save audio files
        sample_rate: sample rate in Hz
        duration: duration of each file in seconds
        format: output format ("wav", "mp3", etc.)

    Returns:
        list of paths to generated audio files
    """
    from .core import save_audio

    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    # Sine waves at different frequencies
    frequencies = [100, 440, 1000, 4000, 8000]
    for freq in frequencies:
        audio, sr = generate_sine_wave(freq, duration, sample_rate)
        filepath = os.path.join(output_dir, f"sine_{freq}hz.{format}")
        save_audio(audio, filepath, sr)
        generated_files.append(filepath)

    # Chirps
    chirp_params = [
        (100, 1000, "linear"),
        (1000, 100, "linear"),
        (100, 8000, "logarithmic"),
    ]
    for freq_start, freq_end, method in chirp_params:
        audio, sr = generate_chirp(
            freq_start, freq_end, duration, sample_rate, method=method
        )
        filepath = os.path.join(
            output_dir, f"chirp_{freq_start}_{freq_end}_{method}.{format}"
        )
        save_audio(audio, filepath, sr)
        generated_files.append(filepath)

    # Noise
    for noise_type in ["white", "pink", "brown"]:
        audio, sr = generate_noise(
            duration, sample_rate, noise_type=noise_type, seed=42
        )
        filepath = os.path.join(output_dir, f"noise_{noise_type}.{format}")
        save_audio(audio, filepath, sr)
        generated_files.append(filepath)

    # Tone bursts
    for onset in [0.2, 0.5, 0.8]:
        audio, sr = generate_tone_burst(
            440, 0.5, duration, onset, sample_rate, envelope="hann"
        )
        filepath = os.path.join(
            output_dir, f"tone_burst_onset_{onset}.{format}"
        )
        save_audio(audio, filepath, sr)
        generated_files.append(filepath)

    # Harmonic tones
    for n_harmonics in [3, 5, 10]:
        audio, sr = generate_harmonic_tone(
            220, duration, sample_rate, n_harmonics=n_harmonics
        )
        filepath = os.path.join(
            output_dir, f"harmonic_220hz_{n_harmonics}harm.{format}"
        )
        save_audio(audio, filepath, sr)
        generated_files.append(filepath)

    # Clicks
    for click_time in [0.25, 0.5, 0.75]:
        audio, sr = generate_click(duration, click_time, sample_rate)
        filepath = os.path.join(output_dir, f"click_{click_time}s.{format}")
        save_audio(audio, filepath, sr)
        generated_files.append(filepath)

    # Combined signals
    # Multi-tone (chord)
    audio = np.zeros(int(duration * sample_rate), dtype=np.float32)
    for freq in [261.63, 329.63, 392.00]:  # C major chord
        tone, _ = generate_sine_wave(
            freq, duration, sample_rate, amplitude=0.3
        )
        audio += tone
    filepath = os.path.join(output_dir, f"chord_c_major.{format}")
    save_audio(audio, filepath, sample_rate)
    generated_files.append(filepath)

    # Signal with multiple events
    audio = np.zeros(int(duration * sample_rate), dtype=np.float32)
    # Add tone burst
    burst, _ = generate_tone_burst(440, 0.3, duration, 0.2, sample_rate)
    audio += burst
    # Add another burst
    burst2, _ = generate_tone_burst(880, 0.3, duration, 1.0, sample_rate)
    audio += burst2
    # Add some noise
    noise, _ = generate_noise(
        duration, sample_rate, amplitude=0.1, noise_type="white", seed=42
    )
    audio += noise
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    filepath = os.path.join(output_dir, f"multi_event.{format}")
    save_audio(audio, filepath, sample_rate)
    generated_files.append(filepath)

    logger.info(
        f"Generated {len(generated_files)} test audio files in {output_dir}"
    )
    return generated_files
