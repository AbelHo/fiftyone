"""
Spectrogram generation utilities.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

import numpy as np

logger = logging.getLogger(__name__)


# Available colormaps for spectrograms
COLORMAP_OPTIONS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "gray",
    "hot",
    "cool",
    "jet",
    "turbo",
]

# Available window functions
WINDOW_OPTIONS = [
    "hann",
    "hamming",
    "blackman",
    "bartlett",
    "kaiser",
    "boxcar",
]


@dataclass
class SpectrogramConfig:
    """Configuration for spectrogram generation.

    Attributes:
        n_fft: FFT window size (number of samples per FFT)
        hop_length: number of samples between successive frames (overlap = n_fft - hop_length)
        win_length: window length for FFT (defaults to n_fft if None)
        window: window function to use
        colormap: matplotlib colormap for visualization
        freq_min: minimum frequency to display (Hz)
        freq_max: maximum frequency to display (Hz)
        db_min: minimum dB value for dynamic range
        db_max: maximum dB value for dynamic range
        ref_db: reference dB level
        power: exponent for magnitude spectrogram (1=amplitude, 2=power)
        mel_scale: whether to use mel scale for frequency axis
        n_mels: number of mel bands (only used if mel_scale=True)
        log_scale: whether to use log scale for frequency axis
        normalize: whether to normalize the spectrogram
        figsize: figure size in inches (width, height)
        dpi: dots per inch for saved images
    """

    n_fft: int = 2048
    hop_length: int = 512
    win_length: Optional[int] = None
    window: str = "hann"
    colormap: str = "viridis"
    freq_min: Optional[float] = None
    freq_max: Optional[float] = None
    db_min: float = -80.0
    db_max: float = 0.0
    ref_db: float = 0.0
    power: float = 2.0
    mel_scale: bool = False
    n_mels: int = 128
    log_scale: bool = False
    normalize: bool = True
    figsize: Tuple[float, float] = (10, 4)
    dpi: int = 100

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.window not in WINDOW_OPTIONS:
            raise ValueError(f"window must be one of {WINDOW_OPTIONS}")
        if self.colormap not in COLORMAP_OPTIONS:
            logger.warning(
                f"Colormap '{self.colormap}' may not be available. "
                f"Standard options are: {COLORMAP_OPTIONS}"
            )
        if self.power <= 0:
            raise ValueError("power must be positive")

    @property
    def overlap(self) -> int:
        """Return the overlap in samples."""
        return self.n_fft - self.hop_length

    @property
    def overlap_ratio(self) -> float:
        """Return the overlap as a ratio (0-1)."""
        return self.overlap / self.n_fft

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": self.window,
            "colormap": self.colormap,
            "freq_min": self.freq_min,
            "freq_max": self.freq_max,
            "db_min": self.db_min,
            "db_max": self.db_max,
            "ref_db": self.ref_db,
            "power": self.power,
            "mel_scale": self.mel_scale,
            "n_mels": self.n_mels,
            "log_scale": self.log_scale,
            "normalize": self.normalize,
            "figsize": self.figsize,
            "dpi": self.dpi,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpectrogramConfig":
        """Create config from dictionary."""
        return cls(**d)


def get_window(window: str, length: int) -> np.ndarray:
    """Get a window function.

    Args:
        window: name of window function
        length: length of window in samples

    Returns:
        window array
    """
    from scipy import signal

    if window == "hann":
        return signal.windows.hann(length)
    elif window == "hamming":
        return signal.windows.hamming(length)
    elif window == "blackman":
        return signal.windows.blackman(length)
    elif window == "bartlett":
        return signal.windows.bartlett(length)
    elif window == "kaiser":
        return signal.windows.kaiser(length, beta=14)
    elif window == "boxcar":
        return signal.windows.boxcar(length)
    else:
        raise ValueError(f"Unknown window function: {window}")


def generate_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    config: Optional[SpectrogramConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a spectrogram from audio data.

    Args:
        audio: audio data as 1D numpy array
        sample_rate: sample rate in Hz
        config: spectrogram configuration

    Returns:
        tuple of (spectrogram, frequencies, times)
            - spectrogram: 2D array of shape (n_frequencies, n_frames) in dB
            - frequencies: 1D array of frequency values
            - times: 1D array of time values
    """
    if config is None:
        config = SpectrogramConfig()

    from scipy import signal

    # Ensure audio is 1D
    if len(audio.shape) > 1:
        audio = np.mean(
            audio, axis=0 if audio.shape[0] > audio.shape[1] else 1
        )

    win_length = config.win_length or config.n_fft
    window = get_window(config.window, win_length)

    if config.mel_scale:
        # Use mel spectrogram
        spec, freqs, times = _compute_mel_spectrogram(
            audio, sample_rate, config, window
        )
    else:
        # Compute standard STFT spectrogram
        freqs, times, Zxx = signal.stft(
            audio,
            fs=sample_rate,
            window=window,
            nperseg=win_length,
            noverlap=config.overlap,
            nfft=config.n_fft,
            return_onesided=True,
        )

        # Convert to power spectrogram
        spec = np.abs(Zxx) ** config.power

    # Convert to dB
    spec_db = 10 * np.log10(spec + 1e-10)

    # Normalize to reference
    spec_db = spec_db - config.ref_db

    # Apply dynamic range limits
    spec_db = np.clip(spec_db, config.db_min, config.db_max)

    if config.normalize:
        # Normalize to 0-1 range
        spec_db = (spec_db - config.db_min) / (config.db_max - config.db_min)

    # Apply frequency limits
    if config.freq_min is not None or config.freq_max is not None:
        freq_min = config.freq_min or 0
        freq_max = config.freq_max or freqs[-1]
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
        spec_db = spec_db[freq_mask, :]
        freqs = freqs[freq_mask]

    return spec_db, freqs, times


def _compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    config: SpectrogramConfig,
    window: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mel spectrogram.

    Args:
        audio: audio data
        sample_rate: sample rate
        config: spectrogram config
        window: window function

    Returns:
        tuple of (mel_spectrogram, mel_frequencies, times)
    """
    from scipy import signal

    # First compute STFT
    win_length = config.win_length or config.n_fft
    freqs, times, Zxx = signal.stft(
        audio,
        fs=sample_rate,
        window=window,
        nperseg=win_length,
        noverlap=config.overlap,
        nfft=config.n_fft,
        return_onesided=True,
    )

    # Compute power spectrogram
    power_spec = np.abs(Zxx) ** config.power

    # Create mel filterbank
    mel_fb = _create_mel_filterbank(
        sample_rate,
        config.n_fft,
        config.n_mels,
        config.freq_min or 0,
        config.freq_max or sample_rate / 2,
    )

    # Apply mel filterbank
    mel_spec = np.dot(mel_fb, power_spec)

    # Create mel frequency axis
    mel_freqs = _hz_to_mel(
        np.linspace(
            config.freq_min or 0,
            config.freq_max or sample_rate / 2,
            config.n_mels,
        )
    )

    return mel_spec, mel_freqs, times


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Convert Hz to mel scale."""
    return 2595 * np.log10(1 + hz / 700)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Convert mel scale to Hz."""
    return 700 * (10 ** (mel / 2595) - 1)


def _create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    freq_min: float,
    freq_max: float,
) -> np.ndarray:
    """Create a mel filterbank matrix.

    Args:
        sample_rate: sample rate
        n_fft: FFT size
        n_mels: number of mel bands
        freq_min: minimum frequency
        freq_max: maximum frequency

    Returns:
        mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)
    """
    # Get mel points
    mel_min = _hz_to_mel(np.array([freq_min]))[0]
    mel_max = _hz_to_mel(np.array([freq_max]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    # Get FFT bins
    n_fft_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_fft_bins)

    # Create filterbank
    filterbank = np.zeros((n_mels, n_fft_bins))

    for i in range(n_mels):
        # Lower slope
        lower = hz_points[i]
        center = hz_points[i + 1]
        upper = hz_points[i + 2]

        for j, freq in enumerate(fft_freqs):
            if lower <= freq < center:
                filterbank[i, j] = (freq - lower) / (center - lower)
            elif center <= freq < upper:
                filterbank[i, j] = (upper - freq) / (upper - center)

    return filterbank


def generate_spectrogram_image(
    audio: np.ndarray,
    sample_rate: int,
    config: Optional[SpectrogramConfig] = None,
    output_path: Optional[str] = None,
    show_colorbar: bool = True,
    show_axes: bool = True,
    title: Optional[str] = None,
) -> np.ndarray:
    """Generate a spectrogram image from audio data.

    Args:
        audio: audio data as 1D numpy array
        sample_rate: sample rate in Hz
        config: spectrogram configuration
        output_path: optional path to save the image
        show_colorbar: whether to show colorbar
        show_axes: whether to show axes
        title: optional title for the plot

    Returns:
        image as numpy array (RGB, shape HxWx3)
    """
    if config is None:
        config = SpectrogramConfig()

    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt

    # Generate spectrogram
    spec_db, freqs, times = generate_spectrogram(audio, sample_rate, config)

    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    # Plot spectrogram
    extent = [times[0], times[-1], freqs[0], freqs[-1]]

    if config.log_scale and not config.mel_scale:
        # Use log scale for frequency axis
        from matplotlib.colors import LogNorm

        im = ax.imshow(
            spec_db,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=config.colormap,
        )
        ax.set_yscale("log")
    else:
        im = ax.imshow(
            spec_db,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=config.colormap,
        )

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        if config.normalize:
            cbar.set_label("Normalized Power")
        else:
            cbar.set_label("Power (dB)")

    if show_axes:
        ax.set_xlabel("Time (s)")
        if config.mel_scale:
            ax.set_ylabel("Mel Frequency")
        else:
            ax.set_ylabel("Frequency (Hz)")
    else:
        ax.axis("off")

    if title:
        ax.set_title(title)

    plt.tight_layout()

    # Convert to image array
    fig.canvas.draw()
    # Use buffer_rgba() for compatibility with newer matplotlib versions
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    # Convert RGBA to RGB
    img = img[:, :, :3]

    if output_path:
        plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")

    plt.close(fig)

    return img


def save_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    output_path: str,
    config: Optional[SpectrogramConfig] = None,
    show_colorbar: bool = False,
    show_axes: bool = False,
    title: Optional[str] = None,
) -> str:
    """Save a spectrogram image to file.

    Args:
        audio: audio data as 1D numpy array
        sample_rate: sample rate in Hz
        output_path: path to save the image
        config: spectrogram configuration
        show_colorbar: whether to show colorbar
        show_axes: whether to show axes
        title: optional title for the plot

    Returns:
        the path to the saved image
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    generate_spectrogram_image(
        audio,
        sample_rate,
        config=config,
        output_path=output_path,
        show_colorbar=show_colorbar,
        show_axes=show_axes,
        title=title,
    )

    return output_path


def spectrogram_to_audio_coords(
    x: float,
    y: float,
    spec_shape: Tuple[int, int],
    duration: float,
    sample_rate: int,
    freq_min: float = 0,
    freq_max: Optional[float] = None,
) -> Tuple[float, float]:
    """Convert spectrogram image coordinates to audio time and frequency.

    Args:
        x: x coordinate in image (0-1, normalized)
        y: y coordinate in image (0-1, normalized)
        spec_shape: shape of spectrogram (height, width)
        duration: audio duration in seconds
        sample_rate: sample rate
        freq_min: minimum frequency
        freq_max: maximum frequency (defaults to Nyquist)

    Returns:
        tuple of (time_seconds, frequency_hz)
    """
    if freq_max is None:
        freq_max = sample_rate / 2

    time_seconds = x * duration
    frequency_hz = freq_min + (1 - y) * (freq_max - freq_min)

    return time_seconds, frequency_hz


def audio_coords_to_spectrogram(
    time_seconds: float,
    frequency_hz: float,
    duration: float,
    sample_rate: int,
    freq_min: float = 0,
    freq_max: Optional[float] = None,
) -> Tuple[float, float]:
    """Convert audio time and frequency to spectrogram image coordinates.

    Args:
        time_seconds: time in seconds
        frequency_hz: frequency in Hz
        duration: audio duration in seconds
        sample_rate: sample rate
        freq_min: minimum frequency
        freq_max: maximum frequency (defaults to Nyquist)

    Returns:
        tuple of (x, y) normalized coordinates (0-1)
    """
    if freq_max is None:
        freq_max = sample_rate / 2

    x = time_seconds / duration
    y = 1 - (frequency_hz - freq_min) / (freq_max - freq_min)

    return x, y
