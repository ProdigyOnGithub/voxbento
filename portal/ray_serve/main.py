from ray import serve

from portal.ray_serve.translator import NLLBTranslator
from portal.ray_serve.transcriber import FasterWhisperTranscriber

app = serve.multiplexed(
    serve.ingress("/translator")(NLLBTranslator).bind(),
    serve.ingress("/transcriber")(FasterWhisperTranscriber).bind(),
)
