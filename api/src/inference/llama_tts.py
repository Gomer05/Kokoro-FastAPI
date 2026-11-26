"""Async client for llama-server's Kokoro TTS API."""

import json
import math
from email.message import Message

import httpx
import numpy as np
from anyio import fail_after

_PROPS_TIMEOUT = 5.0
_TTS_TIMEOUT = 60.0
_MAX_PROPS_BYTES = 1024 * 1024
_MAX_PCM_BYTES = 32 * 1024 * 1024
_PCM_DTYPE = np.dtype("<f4")


class LlamaTTSError(RuntimeError):
    """The sidecar could not complete a TTS operation."""


class LlamaTTS:
    def __init__(self, *, base_url: str):
        invalid_url = "llama-server requires an HTTP(S) base URL without credentials, query, or fragment"
        try:
            url = httpx.URL(base_url)
        except httpx.InvalidURL:
            raise LlamaTTSError(invalid_url) from None
        if (
            url.scheme not in ("http", "https")
            or not url.host
            or url.userinfo
            or "?" in base_url
            or "#" in base_url
        ):
            raise LlamaTTSError(invalid_url)
        self._base_url = str(url).rstrip("/")

    async def _request(
        self,
        method: str,
        path: str,
        *,
        time_limit: float,
        max_bytes: int,
        payload: dict[str, str | float | bool] | None = None,
    ) -> tuple[httpx.Headers, bytearray]:
        operation = f"llama-server {method} {path}"
        try:
            with fail_after(time_limit):
                async with httpx.AsyncClient(
                    follow_redirects=False,
                    trust_env=False,
                    timeout=httpx.Timeout(time_limit, connect=3.0),
                ) as client:
                    async with client.stream(
                        method, self._base_url + path, json=payload
                    ) as response:
                        if response.status_code != 200:
                            raise LlamaTTSError(
                                f"{operation} returned HTTP {response.status_code}; "
                                "check the sidecar URL and llama-server logs"
                            )
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(body) + len(chunk) > max_bytes:
                                raise LlamaTTSError(
                                    f"{operation} returned an unexpectedly large response"
                                )
                            body.extend(chunk)
                        return response.headers, body
        except (TimeoutError, httpx.TimeoutException):
            raise LlamaTTSError(
                f"{operation} timed out; check sidecar availability and load"
            ) from None
        except (httpx.HTTPError, httpx.InvalidURL):
            raise LlamaTTSError(
                f"{operation} failed; check the configured sidecar URL and server health"
            ) from None

    async def voices(self) -> frozenset[str]:
        _, body = await self._request(
            "GET", "/props", time_limit=_PROPS_TIMEOUT, max_bytes=_MAX_PROPS_BYTES
        )
        try:
            props = json.loads(body)
        except (ValueError, RecursionError):
            raise LlamaTTSError("llama-server GET /props did not return JSON") from None

        # /props modalities.audio describes audio input, not audio generation.
        voices = props.get("voices") if isinstance(props, dict) else None
        if (
            not isinstance(voices, list)
            or not voices
            or not all(isinstance(voice, str) and voice.strip() for voice in voices)
        ):
            raise LlamaTTSError(
                "llama-server GET /props did not report valid voices; "
                "check that the sidecar is serving Kokoro with its voice projector"
            )
        return frozenset(voices)

    async def synthesize(self, phonemes: str, voice: str, speed: float) -> np.ndarray:
        if not phonemes or not voice or not math.isfinite(speed) or speed <= 0:
            raise LlamaTTSError(
                "llama-server TTS requires phonemes, a voice, and a finite positive speed"
            )
        headers, body = await self._request(
            "POST",
            "/tts",
            time_limit=_TTS_TIMEOUT,
            max_bytes=_MAX_PCM_BYTES,
            payload={
                "input": phonemes,
                "voice": voice,
                "speed": speed,
                "raw_prompt": True,
                "response_format": "pcm",
                "stream": False,
            },
        )

        content_type = Message()
        content_type["Content-Type"] = headers.get("Content-Type", "")
        if (
            content_type.get_content_type() != "audio/pcm"
            or content_type.get_param("rate") != "24000"
            or content_type.get_param("channels", "1") != "1"
            or content_type.get_param("encoding") != "float"
            or content_type.get_param("bits") != "32"
        ):
            raise LlamaTTSError(
                "llama-server POST /tts did not return mono 24 kHz float32 PCM; "
                "check the sidecar build and model"
            )
        if not body or len(body) % _PCM_DTYPE.itemsize:
            raise LlamaTTSError(
                "llama-server POST /tts returned empty or truncated PCM"
            )

        audio = np.frombuffer(body, dtype=_PCM_DTYPE).astype(np.float32)
        if not np.isfinite(audio).all():
            raise LlamaTTSError(
                "llama-server POST /tts returned non-finite PCM samples"
            )
        return audio
