"""
Fun-CosyVoice3-0.5B backend implementation.

Wraps the vendored CosyVoice3 inference stack for zero-shot voice
cloning. The upstream stack selects CUDA when available and otherwise
falls back to CPU (there is no MPS path), so macOS runs on CPU. Output
is 24 kHz mono audio.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np

# The CosyVoice and Matcha inference sources are vendored under
# backend/vendor (the upstream repo ships no installable package).
_VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

from .base import (  # noqa: E402
    combine_voice_prompts as _combine_voice_prompts,
    empty_device_cache,
    get_torch_device,
    is_model_cached,
    manual_seed,
    model_load_progress,
)

logger = logging.getLogger(__name__)

COSYVOICE3_HF_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
# ModelScope mirrors the same repo under the same id (used as a download
# fallback when the HuggingFace Hub is unreachable).
COSYVOICE3_MS_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

# Files needed for non-streaming inference. The repo also ships a
# TensorRT estimator, a streaming batch speech tokenizer and the RL
# checkpoint; skipping them keeps the download ~5.4 GB instead of ~9.7 GB.
_COSYVOICE3_REQUIRED_FILES = [
    "cosyvoice3.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "CosyVoice-BlankEN/config.json",
    "CosyVoice-BlankEN/tokenizer_config.json",
    "CosyVoice-BlankEN/vocab.json",
    "CosyVoice-BlankEN/merges.txt",
    "CosyVoice-BlankEN/model.safetensors",
]

_COSYVOICE3_IGNORE_PATTERNS = [
    ".gitattributes",
    "asset/*",
    "flow.decoder.estimator.fp32.onnx",
    "llm.rl.pt",
    "speech_tokenizer_v3.batch.onnx",
]


class CosyVoice3TTSBackend:
    """Fun-CosyVoice3-0.5B backend for zero-shot voice cloning."""

    _load_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self):
        self.model = None
        self.model_size = "default"
        self._device = None
        self._model_load_lock = asyncio.Lock()

    def _get_device(self) -> str:
        # Upstream picks CUDA-if-available-else-CPU internally; keep the
        # Voicebox-side device view in sync with what the stack will use.
        return get_torch_device(force_cpu_on_mac=True, allow_xpu=True)

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return COSYVOICE3_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        from ..utils.model_download import is_modelscope_cached

        if is_model_cached(COSYVOICE3_HF_REPO, required_files=_COSYVOICE3_REQUIRED_FILES):
            return True
        return is_modelscope_cached(COSYVOICE3_MS_REPO, required_files=_COSYVOICE3_REQUIRED_FILES)

    async def load_model(self, model_size: str = "default") -> None:
        """Download (if needed) and load the CosyVoice3 model."""
        if self.model is not None:
            return
        async with self._model_load_lock:
            if self.model is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self) -> None:
        model_name = "cosyvoice3"
        is_cached = self._is_model_cached()
        self._device = self._get_device()

        with model_load_progress(model_name, is_cached):
            from ..utils.model_download import download_snapshot

            logger.info("Downloading CosyVoice3 0.5B (this may take a while)...")
            model_dir = download_snapshot(
                COSYVOICE3_HF_REPO,
                ms_repo_id=COSYVOICE3_MS_REPO,
                ignore_patterns=_COSYVOICE3_IGNORE_PATTERNS,
            )
            logger.info("Loading Fun-CosyVoice3-0.5B on %s...", self._device)

            from cosyvoice.cli.cosyvoice import CosyVoice3

            self.model = CosyVoice3(model_dir)

        logger.info("Fun-CosyVoice3-0.5B loaded successfully")

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            empty_device_cache(self._device)
            logger.info("CosyVoice3 unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """
        Create a voice prompt from reference audio.

        CosyVoice3 processes the reference audio at generation time, so
        the prompt just stores the file path plus its transcript.
        """
        voice_prompt = {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }
        return voice_prompt, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate speech with the cloned voice.

        With ``instruct`` set, CosyVoice3 runs in instruct mode
        (instruct text drives the prompt); otherwise the reference
        transcript is used for zero-shot cloning.
        """
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio")
        if not ref_audio or not Path(ref_audio).exists():
            raise ValueError(
                "CosyVoice3 requires a reference audio sample for voice cloning; "
                f"file not found: {ref_audio}"
            )
        ref_text = voice_prompt.get("ref_text", "")

        def _generate_sync():
            import torch

            if seed is not None:
                manual_seed(seed, self._device)

            logger.info("[CosyVoice3] Generating: lang=%s instruct=%s", language, bool(instruct))

            # CosyVoice3 requires the `<|endofprompt|>` marker inside the
            # prompt text (asserted by the LLM forward pass). Without a
            # transcript we fall back to instruct mode with a generic
            # instruction, which still clones from the reference audio.
            if instruct:
                prompt_text = f"You are a helpful assistant. {instruct}<|endofprompt|>"
                generator = self.model.inference_instruct2(
                    text, prompt_text, ref_audio, stream=False
                )
            elif ref_text:
                prompt_text = f"You are a helpful assistant.<|endofprompt|>{ref_text}"
                generator = self.model.inference_zero_shot(
                    text, prompt_text, ref_audio, stream=False
                )
            else:
                prompt_text = "You are a helpful assistant. 请用参考音频的声音朗读以下内容。<|endofprompt|>"
                generator = self.model.inference_instruct2(
                    text, prompt_text, ref_audio, stream=False
                )

            chunks = []
            for output in generator:
                wav = output["tts_speech"]
                if isinstance(wav, torch.Tensor):
                    wav = wav.squeeze().cpu().numpy()
                chunks.append(np.asarray(wav, dtype=np.float32))

            if not chunks:
                raise RuntimeError("CosyVoice3 produced no audio output")
            audio = np.concatenate(chunks)
            return audio, self.model.sample_rate

        return await asyncio.to_thread(_generate_sync)
