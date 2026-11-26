"""Kokoro text processing with synthesis offload to a llama.cpp sidecar."""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Iterator, Optional, Tuple, Union

import numpy as np
import torch
from kokoro import KPipeline
from loguru import logger

from ..core.config import settings
from .base import AudioChunk
from .kokoro_stock_voices import STOCK_VOICE_SHA256, is_stock_voice
from .kokoro_v1 import KokoroV1
from .llama_tts import LlamaTTS


class KokoroVulkan(KokoroV1):
    def __init__(self, url: str):
        super().__init__()
        self._sidecar = LlamaTTS(base_url=url)
        self._sidecar_voices: frozenset[str] | None = None
        self._sidecar_lock = asyncio.Lock()
        self._phonemizers: dict[str, KPipeline] = {}

    async def validate_sidecar(self) -> frozenset[str]:
        if self._sidecar_voices is None:
            async with self._sidecar_lock:
                if self._sidecar_voices is None:
                    voices = await self._sidecar.voices()
                    stock = frozenset(voices & STOCK_VOICE_SHA256.keys())
                    if not stock:
                        raise RuntimeError(
                            "Vulkan sidecar does not report any Kokoro v1.0 voices"
                        )
                    self._sidecar_voices = stock
                    logger.info(
                        f"Connected to the Vulkan sidecar with {len(stock)} stock voices"
                    )
        return self._sidecar_voices

    def _phonemes(self, text: str, lang_code: str) -> Iterator[str]:
        # Lazy initialization here keeps language setup in the worker thread.
        if lang_code not in self._phonemizers:
            self._phonemizers[lang_code] = KPipeline(
                lang_code=lang_code, repo_id=settings.model_repo_id, model=False
            )
        for result in self._phonemizers[lang_code](text):
            yield result.phonemes

    async def _stock_name(
        self, voice: Union[str, Tuple[str, Union[torch.Tensor, str]]]
    ) -> str | None:
        name, path = voice if isinstance(voice, tuple) else (Path(voice).stem, voice)
        if (
            self._sidecar_voices is not None and name not in self._sidecar_voices
        ) or not isinstance(path, str):
            return None
        if await is_stock_voice(name, path) and name in await self.validate_sidecar():
            return name
        return None

    async def generate(
        self,
        text: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        return_timestamps: Optional[bool] = False,
        _retry_oom: bool = True,
    ) -> AsyncGenerator[AudioChunk, None]:
        name = None if return_timestamps else await self._stock_name(voice)
        if name is None:
            async for chunk in super().generate(
                text, voice, speed, lang_code, return_timestamps, _retry_oom
            ):
                yield chunk
            return

        language = lang_code or settings.default_voice_code or name[0].lower()
        async for phonemes in self._iterate_off_loop(self._phonemes(text, language)):
            if phonemes:
                audio = await self._sidecar.synthesize(phonemes, name, speed)
                yield AudioChunk(audio, word_timestamps=None)

    async def generate_from_tokens(
        self,
        tokens: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        _retry_oom: bool = True,
    ) -> AsyncGenerator[np.ndarray, None]:
        if len(tokens) > 510:
            raise ValueError(f"Phoneme string too long: {len(tokens)} > 510")
        # The runtimes select the voice style row on opposite sides of vocab filtering.
        name = None
        if (
            self._model is not None
            and tokens
            and all(p in self._model.vocab for p in tokens)
        ):
            name = await self._stock_name(voice)
        if name is None:
            async for audio in super().generate_from_tokens(
                tokens, voice, speed, lang_code, _retry_oom
            ):
                yield audio
            return
        yield await self._sidecar.synthesize(tokens, name, speed)

    def unload(self) -> None:
        self._sidecar_voices = None
        self._phonemizers.clear()
        super().unload()
