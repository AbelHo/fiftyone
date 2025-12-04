"""
Audio playback operator.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types


class PlayAudio(foo.Operator):
    """Operator to play audio from a sample."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="play_audio",
            label="Play Audio",
            description="Play the audio file associated with a sample",
            icon="play_arrow",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        # Get sample ID
        sample_id = ctx.params.get("sample_id")
        if sample_id is None and ctx.current_sample:
            sample_id = ctx.current_sample

        if sample_id:
            sample = ctx.dataset[sample_id]
            audio_path = sample.get("audio_path")

            if audio_path:
                # Create media player view
                inputs.view(
                    "player",
                    types.MediaPlayerView(
                        label="Audio Player",
                        description="Click play to listen to the audio",
                    ),
                )
                inputs.str(
                    "audio_url", default=audio_path, view=types.HiddenView()
                )
            else:
                inputs.view(
                    "message",
                    types.Notice(
                        label="No Audio",
                        description="This sample does not have an associated audio file",
                    ),
                )
        else:
            inputs.view(
                "message",
                types.Notice(
                    label="No Sample Selected",
                    description="Please select a sample to play its audio",
                ),
            )

        return types.Property(inputs)

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("status", label="Status")
        return types.Property(outputs)

    def execute(self, ctx):
        sample_id = ctx.params.get("sample_id")
        if sample_id is None and ctx.current_sample:
            sample_id = ctx.current_sample

        if sample_id:
            sample = ctx.dataset[sample_id]
            audio_path = sample.get("audio_path")
            if audio_path:
                return {"status": "playing", "audio_path": audio_path}

        return {"status": "no_audio"}


class GetAudioInfo(foo.Operator):
    """Operator to get audio information for a sample."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="get_audio_info",
            label="Get Audio Info",
            description="Get audio metadata for a sample",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("sample_id", label="Sample ID", required=True)
        return types.Property(inputs)

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.obj("audio_info", label="Audio Information")
        return types.Property(outputs)

    def execute(self, ctx):
        sample_id = ctx.params.get("sample_id")

        if sample_id is None:
            return {"audio_info": None}

        sample = ctx.dataset[sample_id]
        audio_path = sample.get("audio_path")

        if audio_path is None:
            return {"audio_info": None}

        # Get audio metadata
        audio_metadata = sample.get("audio_metadata", {})

        return {
            "audio_info": {
                "audio_path": audio_path,
                "duration": audio_metadata.get("duration"),
                "sample_rate": audio_metadata.get("sample_rate"),
                "channels": audio_metadata.get("channels"),
                "encoding": audio_metadata.get("encoding"),
            }
        }


def register(p):
    p.register(PlayAudio)
    p.register(GetAudioInfo)
