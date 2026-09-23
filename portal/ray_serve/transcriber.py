"""Ray Serve deployment for Whisper audio transcription."""

import base64
import logging

import numpy as np
from ray import serve
from starlette.requests import Request

logger = logging.getLogger(__name__)


@serve.deployment
class FasterWhisperTranscriber:
    """Ray Serve deployment class for the Faster Whisper transcription model."""

    def __init__(self, model_size: str = "tiny"):
        """
        Initialize the Whisper model with dynamic hardware detection.

        Args:
            model_size: The model size (e.g., 'tiny', 'base', 'large-v3').
        """
        from faster_whisper import WhisperModel

        self.model_size = model_size
        logger.info(f"Loading faster-whisper model: {model_size}")

        import ray

        has_gpu = len(ray.get_gpu_ids()) > 0
        device = "cuda" if has_gpu else "cpu"
        compute_type = "float16" if has_gpu else "int8"

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info(f"Whisper model {model_size} loaded successfully.")

    def transcribe(self, audio_data: np.ndarray, language_code: str) -> str:
        """
        Run the audio data through the transcription model.

        Args:
            audio_data: Numpy array of the float32 audio samples.
            language_code: Optional ISO language code to force the model into.

        Returns:
            The transcribed text string.
        """
        segments, _ = self.model.transcribe(
            audio_data,
            beam_size=5,
            vad_filter=True,
            language=language_code if language_code else None,
            word_timestamps=True,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            condition_on_previous_text=False,
        )

        valid_words = []
        for segment in segments:
            if not getattr(segment, "words", None):
                # Fallback if words aren't available
                valid_words.append(segment.text.strip())
                continue

            for word in segment.words:
                if word.end > 1.0:  # Skip words completely inside the 1.0s overlap period
                    valid_words.append(word.word.strip())

        return " ".join(valid_words).strip()

    async def __call__(self, request: Request):
        """
        Handle incoming HTTP requests to the deployment.

        Args:
            request: The Starlette HTTP request containing the JSON payload.

        Returns:
            A dictionary containing the transcribed text or an error.
        """
        try:
            payload = await request.json()
        except ValueError:
            return {"error": "Invalid JSON payload."}

        if not isinstance(payload, dict):
            return {"error": "Expected JSON dictionary payload."}

        audio_b64 = payload.get("audio_b64")
        if not audio_b64:
            return {"error": "Missing audio_b64 in payload."}

        # Convert back from base64 to bytes, then to numpy float32
        audio_bytes = base64.b64decode(audio_b64)
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)

        language_code = payload.get("language_code", "")

        import asyncio

        loop = asyncio.get_running_loop()
        # Run inference in background thread so we don't block the Ray event loop
        transcribed_text = await loop.run_in_executor(None, lambda: self.transcribe(audio_data, language_code))

        return {"transcribed_text": transcribed_text}


transcriber_app = FasterWhisperTranscriber.bind()
