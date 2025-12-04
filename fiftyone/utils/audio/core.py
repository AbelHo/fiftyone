"""
Core audio processing utilities.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import os
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def load_audio(
    filepath: str,
    sr: Optional[int] = None,
    mono: bool = True,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Load an audio file.

    Args:
        filepath: path to the audio file
        sr: target sample rate. If None, uses the native sample rate
        mono: whether to convert to mono
        offset: start time in seconds
        duration: duration in seconds to load. If None, loads entire file

    Returns:
        tuple of (audio_data, sample_rate)
            - audio_data: numpy array of shape (samples,) for mono or (channels, samples) for stereo
            - sample_rate: the sample rate of the audio

    Raises:
        FileNotFoundError: if the audio file does not exist
        ValueError: if the audio file cannot be loaded
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    # Try scipy.io.wavfile first for WAV files
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == ".wav":
            audio, sample_rate = _load_wav(
                filepath, sr, mono, offset, duration
            )
        else:
            # Try soundfile for other formats
            audio, sample_rate = _load_soundfile(
                filepath, sr, mono, offset, duration
            )
    except Exception as e:
        # Fallback to librosa if available
        try:
            audio, sample_rate = _load_librosa(
                filepath, sr, mono, offset, duration
            )
        except ImportError:
            raise ValueError(
                f"Could not load audio file {filepath}. "
                f"Install librosa for broader format support: pip install librosa"
            ) from e

    return audio, sample_rate


def _load_wav(
    filepath: str,
    sr: Optional[int],
    mono: bool,
    offset: float,
    duration: Optional[float],
) -> Tuple[np.ndarray, int]:
    """Load a WAV file using scipy."""
    from scipy.io import wavfile

    sample_rate, audio = wavfile.read(filepath)

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Handle offset and duration
    if offset > 0:
        start_sample = int(offset * sample_rate)
        audio = audio[start_sample:]

    if duration is not None:
        end_sample = int(duration * sample_rate)
        audio = audio[:end_sample]

    # Convert to mono if needed
    if mono and len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr is not None and sr != sample_rate:
        audio = _resample(audio, sample_rate, sr)
        sample_rate = sr

    return audio, sample_rate


def _load_soundfile(
    filepath: str,
    sr: Optional[int],
    mono: bool,
    offset: float,
    duration: Optional[float],
) -> Tuple[np.ndarray, int]:
    """Load an audio file using soundfile."""
    import soundfile as sf

    info = sf.info(filepath)
    sample_rate = info.samplerate

    # Calculate frame range
    start_frame = int(offset * sample_rate)
    if duration is not None:
        frames = int(duration * sample_rate)
    else:
        frames = -1

    audio, sample_rate = sf.read(filepath, start=start_frame, frames=frames)

    # Convert to mono if needed
    if mono and len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Ensure float32
    audio = audio.astype(np.float32)

    # Resample if needed
    if sr is not None and sr != sample_rate:
        audio = _resample(audio, sample_rate, sr)
        sample_rate = sr

    return audio, sample_rate


def _load_librosa(
    filepath: str,
    sr: Optional[int],
    mono: bool,
    offset: float,
    duration: Optional[float],
) -> Tuple[np.ndarray, int]:
    """Load an audio file using librosa."""
    import librosa

    audio, sample_rate = librosa.load(
        filepath,
        sr=sr,
        mono=mono,
        offset=offset,
        duration=duration,
    )

    return audio.astype(np.float32), sample_rate


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to a target sample rate."""
    from scipy import signal

    num_samples = int(len(audio) * target_sr / orig_sr)
    resampled = signal.resample(audio, num_samples)
    return resampled.astype(np.float32)


def get_audio_info(filepath: str) -> dict:
    """Get information about an audio file.

    Args:
        filepath: path to the audio file

    Returns:
        dict containing:
            - sample_rate: sample rate in Hz
            - channels: number of channels
            - duration: duration in seconds
            - samples: total number of samples
            - format: audio format/extension
            - size_bytes: file size in bytes

    Raises:
        FileNotFoundError: if the audio file does not exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    size_bytes = os.path.getsize(filepath)

    try:
        if ext == ".wav":
            from scipy.io import wavfile

            sample_rate, audio = wavfile.read(filepath)
            if len(audio.shape) == 1:
                channels = 1
                samples = len(audio)
            else:
                channels = audio.shape[1]
                samples = audio.shape[0]
            duration = samples / sample_rate
        else:
            try:
                import soundfile as sf

                info = sf.info(filepath)
                sample_rate = info.samplerate
                channels = info.channels
                duration = info.duration
                samples = info.frames
            except ImportError:
                import librosa

                audio, sample_rate = librosa.load(
                    filepath, sr=None, mono=False
                )
                if len(audio.shape) == 1:
                    channels = 1
                    samples = len(audio)
                else:
                    channels = audio.shape[0]
                    samples = audio.shape[1]
                duration = samples / sample_rate

    except Exception as e:
        raise ValueError(f"Could not get info for audio file {filepath}: {e}")

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "duration": duration,
        "samples": samples,
        "format": ext.lstrip("."),
        "size_bytes": size_bytes,
    }


def normalize_audio(
    audio: np.ndarray,
    method: str = "peak",
    target_level: float = 1.0,
) -> np.ndarray:
    """Normalize audio data.

    Args:
        audio: audio data as numpy array
        method: normalization method, one of:
            - "peak": normalize to peak amplitude
            - "rms": normalize to target RMS level
            - "lufs": loudness normalization (requires pyloudnorm)
        target_level: target level for normalization

    Returns:
        normalized audio data
    """
    audio = audio.astype(np.float32)

    if method == "peak":
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (target_level / peak)

    elif method == "rms":
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio * (target_level / rms)

    elif method == "lufs":
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(rate=44100)  # Assumes 44.1kHz
            loudness = meter.integrated_loudness(audio)
            audio = pyln.normalize.loudness(audio, loudness, target_level)
        except ImportError:
            logger.warning(
                "pyloudnorm not installed, falling back to peak normalization"
            )
            return normalize_audio(
                audio, method="peak", target_level=target_level
            )

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return audio


def save_audio(
    audio: np.ndarray,
    filepath: str,
    sample_rate: int,
    format: Optional[str] = None,
) -> str:
    """Save audio data to a file.

    Args:
        audio: audio data as numpy array
        filepath: output file path
        sample_rate: sample rate in Hz
        format: output format. If None, inferred from filepath extension

    Returns:
        the filepath of the saved audio file
    """
    ext = format or os.path.splitext(filepath)[1].lower().lstrip(".")

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    if ext == "wav":
        from scipy.io import wavfile

        # Convert to int16 for WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(filepath, sample_rate, audio_int16)
    else:
        try:
            import soundfile as sf

            sf.write(filepath, audio, sample_rate)
        except ImportError:
            raise ValueError(
                f"Cannot save to {ext} format. Install soundfile: pip install soundfile"
            )

    return filepath
