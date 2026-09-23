"""Ray Serve deployment for NLLB text translation."""

import asyncio
import logging
import os

from ray import serve
from starlette.requests import Request

logger = logging.getLogger(__name__)


def get_hf_repo_and_revision(model_size: str) -> tuple[str, str]:
    hf_repo_id = model_size
    rev = "main"
    if model_size == "nllb-200-distilled-600M":
        hf_repo_id = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
        rev = "302d78f00e6fdb50a1064059df7c392b735e9d05"
    return hf_repo_id, rev


@serve.deployment
class NLLBTranslator:
    """Ray Serve deployment class for the NLLB translation model."""

    def __init__(self, model_size: str = "nllb-200-distilled-600M"):
        """
        Initialize the NLLB model with dynamic hardware detection.

        Args:
            model_size: The HuggingFace repo ID or local path for the NLLB model.
        """
        import ctranslate2
        import transformers
        from huggingface_hub import snapshot_download

        self.model_size = model_size
        logger.info(f"Loading NLLB model: {model_size}")

        if not os.path.exists(model_size):
            try:
                hf_repo_id, rev = get_hf_repo_and_revision(model_size)
                self.local_model_path = snapshot_download(repo_id=hf_repo_id, revision=rev)
            except Exception as e:
                logger.error(f"Failed to download {model_size} from HuggingFace: {e}")
                self.local_model_path = model_size
        else:
            self.local_model_path = model_size

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.local_model_path, src_lang="eng_Latn", revision="main"
        )

        import ray

        has_gpu = len(ray.get_gpu_ids()) > 0
        device = "cuda" if has_gpu else "cpu"
        compute_type = "float16" if has_gpu else "int8"

        assigned_cpus = int(ray.get_runtime_context().get_assigned_resources().get("CPU", 2))
        intra_threads = max(1, assigned_cpus)

        self.model = ctranslate2.Translator(
            self.local_model_path,
            device=device,
            compute_type=compute_type,
            inter_threads=1,
            intra_threads=intra_threads,
        )
        logger.info(f"NLLB model {model_size} loaded successfully.")

    @serve.batch(max_batch_size=20, batch_wait_timeout_s=0.05)
    async def translate_batch(self, requests: list[dict]) -> list[str]:
        """
        Process a batch of translation requests concurrently.

        Args:
            requests: A list of dictionaries containing text and language tokens.

        Returns:
            A list of translated text strings.
        """
        if not requests:
            return []

        sources = []
        target_prefixes = []

        for req in requests:
            text = req.get("text", "")
            source_lang_token = req.get("source_lang_token")
            target_lang_token = req.get("target_lang_token")

            if not text.strip():
                sources.append([])
                target_prefixes.append([])
                continue

            # Encode text
            source = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text.strip()))

            # NLLB expects the source lang token as the first token
            if source and source_lang_token:
                source[0] = source_lang_token

            sources.append(source)
            target_prefixes.append([target_lang_token])

        # Extract valid requests to avoid ctranslate2 crashing on empty lists
        valid_indices = [i for i, src in enumerate(sources) if src]
        valid_sources = [sources[i] for i in valid_indices]
        valid_prefixes = [target_prefixes[i] for i in valid_indices]

        results = [None] * len(requests)

        if valid_sources:
            loop = asyncio.get_running_loop()
            batch_results = await loop.run_in_executor(
                None,
                lambda: self.model.translate_batch(
                    valid_sources,
                    target_prefix=valid_prefixes,
                    beam_size=1,
                    max_decoding_length=256,
                ),
            )

            for i, result in zip(valid_indices, batch_results):
                results[i] = result

        # Decode results
        translated_texts = []
        for i, result in enumerate(results):
            if not result or not result.hypotheses:
                translated_texts.append("")
                continue

            target = result.hypotheses[0]
            target_lang_token = requests[i].get("target_lang_token")

            # Strip target language prefix token if present
            if target and target[0] == target_lang_token:
                target = target[1:]

            translated_text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target))
            translated_texts.append(translated_text.strip())

        return translated_texts

    async def __call__(self, request: Request):
        """
        Handle incoming HTTP requests to the deployment.

        Args:
            request: The Starlette HTTP request containing the JSON payload.

        Returns:
            A dictionary containing the translated text or an error.
        """
        payload = await request.json()

        if isinstance(payload, dict):
            translated_text = await self.translate_batch(payload)
            return {"translated_text": translated_text}

        return {"error": "Expected JSON dictionary payload."}


translator_app = NLLBTranslator.bind()
