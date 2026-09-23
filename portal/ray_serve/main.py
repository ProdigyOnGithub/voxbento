"""Entrypoint for the Ray Serve multiplexed deployments."""

from ray import serve

from portal.ray_serve.transcriber import FasterWhisperTranscriber
from portal.ray_serve.translator import NLLBTranslator

app = serve.multiplexed(
    serve.ingress("/translator")(NLLBTranslator).bind(),
    serve.ingress("/transcriber")(FasterWhisperTranscriber).bind(),
)
