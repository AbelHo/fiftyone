"""
Spectrogram generation and audio analysis operators.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types


class GenerateSpectrogram(foo.Operator):
    """Operator to generate a spectrogram for a single audio file."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="generate_spectrogram",
            label="Generate Spectrogram",
            description="Generate a spectrogram image from an audio file",
            icon="graphic_eq",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        inputs.str(
            "audio_path",
            label="Audio File Path",
            description="Path to the audio file",
            required=True,
        )

        inputs.str(
            "output_path",
            label="Output Path",
            description="Path to save the spectrogram image (optional)",
        )

        # Spectrogram parameters
        params = inputs.obj(
            "spectrogram_params", label="Spectrogram Parameters"
        )

        params.int(
            "n_fft",
            label="FFT Size",
            description="Number of samples per FFT window",
            default=2048,
        )

        params.int(
            "hop_length",
            label="Hop Length",
            description="Number of samples between frames (lower = more overlap)",
            default=512,
        )

        params.enum(
            "colormap",
            label="Color Map",
            description="Color scheme for the spectrogram",
            values=[
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
            ],
            default="viridis",
        )

        params.enum(
            "window",
            label="Window Function",
            values=[
                "hann",
                "hamming",
                "blackman",
                "bartlett",
                "kaiser",
                "boxcar",
            ],
            default="hann",
        )

        params.float(
            "freq_min",
            label="Min Frequency (Hz)",
            description="Minimum frequency to display",
        )

        params.float(
            "freq_max",
            label="Max Frequency (Hz)",
            description="Maximum frequency to display",
        )

        params.float(
            "db_min",
            label="Min dB",
            description="Minimum dB level for dynamic range",
            default=-80.0,
        )

        params.float(
            "db_max",
            label="Max dB",
            description="Maximum dB level for dynamic range",
            default=0.0,
        )

        params.bool(
            "mel_scale",
            label="Use Mel Scale",
            description="Use mel frequency scale",
            default=False,
        )

        return types.Property(inputs)

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("spectrogram_path", label="Spectrogram Path")
        outputs.bool("success", label="Success")
        return types.Property(outputs)

    def execute(self, ctx):
        from fiftyone.utils.audio import (
            load_audio,
            save_spectrogram,
            SpectrogramConfig,
        )

        audio_path = ctx.params.get("audio_path")
        output_path = ctx.params.get("output_path")
        params = ctx.params.get("spectrogram_params", {})

        if output_path is None:
            # Generate output path
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_dir = os.path.join(
                os.path.dirname(audio_path), "spectrograms"
            )
            output_path = os.path.join(
                output_dir, f"{audio_name}_spectrogram.png"
            )

        try:
            # Load audio
            audio, sr = load_audio(audio_path)

            # Create config
            config = SpectrogramConfig(
                n_fft=params.get("n_fft", 2048),
                hop_length=params.get("hop_length", 512),
                window=params.get("window", "hann"),
                colormap=params.get("colormap", "viridis"),
                freq_min=params.get("freq_min"),
                freq_max=params.get("freq_max"),
                db_min=params.get("db_min", -80.0),
                db_max=params.get("db_max", 0.0),
                mel_scale=params.get("mel_scale", False),
            )

            # Generate spectrogram
            save_spectrogram(audio, sr, output_path, config=config)

            return {"spectrogram_path": output_path, "success": True}

        except Exception as e:
            return {
                "spectrogram_path": None,
                "success": False,
                "error": str(e),
            }


class RegenerateSpectrograms(foo.Operator):
    """Operator to regenerate spectrograms for all samples in a dataset."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="regenerate_spectrograms",
            label="Regenerate Spectrograms",
            description="Regenerate spectrogram images for all audio samples",
            icon="refresh",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        inputs.str(
            "output_dir",
            label="Output Directory",
            description="Directory to save spectrograms (optional)",
        )

        # Spectrogram parameters
        params = inputs.obj(
            "spectrogram_params", label="Spectrogram Parameters"
        )

        params.int("n_fft", label="FFT Size", default=2048)
        params.int("hop_length", label="Hop Length", default=512)
        params.enum(
            "colormap",
            label="Color Map",
            values=[
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
            ],
            default="viridis",
        )
        params.bool("mel_scale", label="Use Mel Scale", default=False)

        inputs.bool(
            "overwrite",
            label="Overwrite Existing",
            description="Overwrite existing spectrogram images",
            default=False,
        )

        inputs.bool(
            "update_filepaths",
            label="Update Filepaths",
            description="Update sample filepaths to point to spectrograms",
            default=False,
        )

        return types.Property(inputs)

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.int("processed", label="Samples Processed")
        outputs.int("failed", label="Samples Failed")
        return types.Property(outputs)

    def execute(self, ctx):
        from fiftyone.utils.audio import (
            generate_spectrograms_for_dataset,
            update_dataset_filepaths_to_spectrograms,
            SpectrogramConfig,
        )

        output_dir = ctx.params.get("output_dir")
        params = ctx.params.get("spectrogram_params", {})
        overwrite = ctx.params.get("overwrite", False)
        update_filepaths = ctx.params.get("update_filepaths", False)

        config = SpectrogramConfig(
            n_fft=params.get("n_fft", 2048),
            hop_length=params.get("hop_length", 512),
            colormap=params.get("colormap", "viridis"),
            mel_scale=params.get("mel_scale", False),
        )

        processed = 0
        failed = 0

        try:
            generate_spectrograms_for_dataset(
                ctx.dataset,
                spectrogram_dir=output_dir,
                spectrogram_config=config,
                overwrite=overwrite,
            )
            processed = len(ctx.dataset)

            if update_filepaths:
                update_dataset_filepaths_to_spectrograms(ctx.dataset)

        except Exception as e:
            failed = len(ctx.dataset)
            return {"processed": 0, "failed": failed, "error": str(e)}

        return {"processed": processed, "failed": failed}


class RunAudioDetection(foo.Operator):
    """Operator to run audio detection on the dataset."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="run_audio_detection",
            label="Run Audio Detection",
            description="Run audio event detection on all samples",
            icon="search",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        inputs.enum(
            "detector_type",
            label="Detector Type",
            values=["threshold", "spectral_peak"],
            default="threshold",
        )

        inputs.str(
            "output_field",
            label="Output Field",
            description="Field to store detections",
            default="detections",
        )

        # Threshold detector params
        threshold_params = inputs.obj(
            "threshold_params", label="Threshold Detector Parameters"
        )
        threshold_params.float(
            "threshold", label="Amplitude Threshold", default=0.1
        )
        threshold_params.float(
            "min_duration", label="Min Duration (s)", default=0.1
        )
        threshold_params.str("label", label="Detection Label", default="sound")

        # Spectral peak detector params
        spectral_params = inputs.obj(
            "spectral_params", label="Spectral Peak Detector Parameters"
        )
        spectral_params.float(
            "threshold_db", label="Threshold (dB)", default=-40
        )
        spectral_params.float(
            "min_duration", label="Min Duration (s)", default=0.05
        )
        spectral_params.float(
            "freq_min", label="Min Frequency (Hz)", default=100
        )
        spectral_params.float(
            "freq_max", label="Max Frequency (Hz)", default=8000
        )

        return types.Property(inputs)

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.int("detections_count", label="Total Detections")
        outputs.int("samples_processed", label="Samples Processed")
        return types.Property(outputs)

    def execute(self, ctx):
        from fiftyone.utils.audio import (
            ThresholdDetector,
            SpectralPeakDetector,
            run_detection,
        )

        detector_type = ctx.params.get("detector_type", "threshold")
        output_field = ctx.params.get("output_field", "detections")

        if detector_type == "threshold":
            params = ctx.params.get("threshold_params", {})
            detector = ThresholdDetector(
                threshold=params.get("threshold", 0.1),
                min_duration=params.get("min_duration", 0.1),
                label=params.get("label", "sound"),
            )
        else:
            params = ctx.params.get("spectral_params", {})
            detector = SpectralPeakDetector(
                threshold_db=params.get("threshold_db", -40),
                min_duration=params.get("min_duration", 0.05),
                freq_min=params.get("freq_min", 100),
                freq_max=params.get("freq_max", 8000),
            )

        run_detection(ctx.dataset, detector, output_field=output_field)

        # Count detections
        total_detections = 0
        samples_processed = 0
        for sample in ctx.dataset:
            dets = sample.get(output_field)
            if dets is not None:
                total_detections += len(dets.detections)
                samples_processed += 1

        return {
            "detections_count": total_detections,
            "samples_processed": samples_processed,
        }


class ComputeAudioSimilarity(foo.Operator):
    """Operator to compute audio similarity scores."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_audio_similarity",
            label="Compute Audio Similarity",
            description="Compute FFT-based similarity scores for audio samples",
            icon="compare",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        inputs.str(
            "reference_sample_id",
            label="Reference Sample ID",
            description="Sample to compare against (leave empty for mean comparison)",
        )

        inputs.int(
            "n_fft",
            label="FFT Size",
            default=2048,
        )

        inputs.int(
            "n_features",
            label="Number of Features",
            default=128,
        )

        inputs.enum(
            "similarity_method",
            label="Similarity Method",
            values=["cosine", "euclidean", "correlation"],
            default="cosine",
        )

        inputs.str(
            "features_field",
            label="Features Field",
            default="fft_features",
        )

        inputs.str(
            "similarity_field",
            label="Similarity Field",
            default="similarity",
        )

        return types.Property(inputs)

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.float("mean_similarity", label="Mean Similarity")
        outputs.float("max_similarity", label="Max Similarity")
        outputs.float("min_similarity", label="Min Similarity")
        return types.Property(outputs)

    def execute(self, ctx):
        from fiftyone.utils.audio import (
            compute_dataset_fft_features,
            compute_dataset_similarity,
        )

        reference_id = ctx.params.get("reference_sample_id")
        n_fft = ctx.params.get("n_fft", 2048)
        n_features = ctx.params.get("n_features", 128)
        method = ctx.params.get("similarity_method", "cosine")
        features_field = ctx.params.get("features_field", "fft_features")
        similarity_field = ctx.params.get("similarity_field", "similarity")

        # Compute FFT features
        compute_dataset_fft_features(
            ctx.dataset,
            output_field=features_field,
            n_fft=n_fft,
            n_features=n_features,
        )

        # Compute similarity
        compute_dataset_similarity(
            ctx.dataset,
            reference_sample_id=reference_id if reference_id else None,
            features_field=features_field,
            output_field=similarity_field,
            method=method,
        )

        # Get statistics
        similarities = []
        for sample in ctx.dataset:
            sim = sample.get(similarity_field)
            if sim is not None:
                similarities.append(sim)

        if similarities:
            return {
                "mean_similarity": sum(similarities) / len(similarities),
                "max_similarity": max(similarities),
                "min_similarity": min(similarities),
            }

        return {
            "mean_similarity": 0,
            "max_similarity": 0,
            "min_similarity": 0,
        }


def register(p):
    p.register(GenerateSpectrogram)
    p.register(RegenerateSpectrograms)
    p.register(RunAudioDetection)
    p.register(ComputeAudioSimilarity)
