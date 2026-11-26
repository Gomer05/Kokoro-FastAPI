from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from api.src.core.config import settings
from api.src.inference.base import AudioChunk
from api.src.inference.kokoro_v1 import KokoroV1
from api.src.inference.kokoro_vulkan import KokoroVulkan
from api.src.inference.model_manager import ModelManager

STOCK_PACK = str(Path(__file__).resolve().parents[1] / "src/voices/v1_0/af_heart.pt")


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(settings, "use_gpu", False)
    instance = KokoroVulkan("http://llama:8080")
    instance._model = SimpleNamespace(vocab=set("ɔla"))
    instance._sidecar = AsyncMock()
    instance._sidecar.voices.return_value = frozenset({"af_heart", "vendor_voice"})
    instance._sidecar.synthesize.return_value = np.ones(3, dtype=np.float32)
    instance._sidecar_voices = frozenset({"af_heart"})
    return instance


@pytest.mark.parametrize("url", [None, "http://llama:8080"])
async def test_model_manager_selects_vulkan_only_when_configured(monkeypatch, url):
    monkeypatch.setattr(settings, "vulkan_tts_url", url)
    monkeypatch.setattr(settings, "use_gpu", False)
    manager = ModelManager()

    await manager.initialize()

    assert type(manager.get_backend()) is (KokoroVulkan if url else KokoroV1)
    assert manager.status()["device"] == ("vulkan" if url else "cpu")
    assert manager.current_backend == ("kokoro_vulkan" if url else "kokoro_v1")
    assert manager.get_backend().device == "cpu"


async def test_validate_sidecar_caches_stock_voices(backend):
    backend._sidecar_voices = None

    assert await backend.validate_sidecar() == {"af_heart"}
    assert await backend.validate_sidecar() == {"af_heart"}

    backend._sidecar.voices.assert_awaited_once()


async def test_validate_sidecar_rejects_missing_kokoro_voices(backend):
    backend._sidecar_voices = None
    backend._sidecar.voices.return_value = frozenset({"vendor_voice"})

    with pytest.raises(RuntimeError, match="does not report any Kokoro"):
        await backend.validate_sidecar()


async def test_stock_voice_uses_existing_phonemes_and_sends_speed(backend, monkeypatch):
    pipeline = Mock(
        return_value=iter(
            [SimpleNamespace(phonemes=phonemes) for phonemes in ("ɔla", "", "mˈundo")]
        )
    )
    create_pipeline = Mock(return_value=pipeline)
    monkeypatch.setattr("api.src.inference.kokoro_vulkan.KPipeline", create_pipeline)

    chunks = [
        chunk
        async for chunk in backend.generate(
            "Hola, mundo", ("af_heart", STOCK_PACK), speed=1.5, lang_code="e"
        )
    ]

    create_pipeline.assert_called_once_with(
        lang_code="e", repo_id=settings.model_repo_id, model=False
    )
    pipeline.assert_called_once_with("Hola, mundo")
    assert [call.args for call in backend._sidecar.synthesize.await_args_list] == [
        ("ɔla", "af_heart", 1.5),
        ("mˈundo", "af_heart", 1.5),
    ]
    assert len(chunks) == 2
    assert all(chunk.audio.dtype == np.float32 for chunk in chunks)
    assert all(chunk.word_timestamps is None for chunk in chunks)


@pytest.mark.parametrize("kind", ["timestamps", "same_name_override", "mix"])
async def test_unsupported_requests_use_existing_torch_path(
    backend, monkeypatch, tmp_path, kind
):
    route = ("af_heart", STOCK_PACK)
    if kind == "same_name_override":
        replacement = tmp_path / "af_heart.pt"
        replacement.write_bytes(b"custom voice")
        route = ("af_heart", str(replacement))
    elif kind == "mix":
        route = ("af_heart+af_bella", STOCK_PACK)

    expected = AudioChunk(np.zeros(2, dtype=np.float32))

    async def torch_chunks(*args, **kwargs):
        yield expected

    torch_generate = Mock(side_effect=torch_chunks)
    monkeypatch.setattr(KokoroV1, "generate", torch_generate)

    chunks = [
        chunk
        async for chunk in backend.generate(
            "example", route, return_timestamps=kind == "timestamps"
        )
    ]

    assert chunks == [expected]
    torch_generate.assert_called_once_with(
        "example", route, 1.0, None, kind == "timestamps", True
    )
    backend._sidecar.synthesize.assert_not_awaited()


async def test_raw_phonemes_use_sidecar_and_preserve_length_limit(backend):
    voice = ("af_heart", STOCK_PACK)
    chunks = [chunk async for chunk in backend.generate_from_tokens("ɔla", voice)]
    backend._sidecar.synthesize.assert_awaited_once_with("ɔla", "af_heart", 1.0)
    assert len(chunks) == 1

    with pytest.raises(ValueError, match="Phoneme string too long"):
        async for _ in backend.generate_from_tokens("a" * 511, voice):
            pass


@pytest.mark.parametrize(
    ("phonemes", "voice"),
    [
        ("ɔla☃", ("af_heart", STOCK_PACK)),
        ("☃", ("af_heart", STOCK_PACK)),
        ("ɔla", ("af_heart+af_bella", STOCK_PACK)),
    ],
)
async def test_raw_custom_voices_and_unknown_phonemes_use_torch(
    backend, monkeypatch, phonemes, voice
):
    expected = np.ones(3, dtype=np.float32)

    async def torch_chunks(*args):
        yield expected

    generate = Mock(side_effect=torch_chunks)
    monkeypatch.setattr(KokoroV1, "generate_from_tokens", generate)

    chunks = [chunk async for chunk in backend.generate_from_tokens(phonemes, voice)]

    assert chunks == [expected]
    generate.assert_called_once()
    backend._sidecar.synthesize.assert_not_awaited()


async def test_sidecar_outage_does_not_block_torch_after_reload(monkeypatch):
    monkeypatch.setattr(settings, "use_gpu", False)
    monkeypatch.setattr(settings, "vulkan_tts_url", "http://llama:8080")
    props = AsyncMock(side_effect=RuntimeError("sidecar unavailable"))
    monkeypatch.setattr("api.src.inference.llama_tts.LlamaTTS.voices", props)
    torch_load = AsyncMock()
    monkeypatch.setattr(KokoroV1, "load_model", torch_load)
    expected = AudioChunk(np.zeros(3, dtype=np.float32))

    async def torch_chunks(*args):
        yield expected

    monkeypatch.setattr(KokoroV1, "generate", torch_chunks)
    manager = ModelManager()
    await manager.initialize()
    await manager.unload()
    voice = ("af_heart", STOCK_PACK)

    chunks = [
        chunk
        async for chunk in manager.generate("caption", voice, return_timestamps=True)
    ]

    assert chunks == [expected]
    torch_load.assert_awaited_once()
    props.assert_not_awaited()

    with pytest.raises(RuntimeError, match="sidecar unavailable"):
        async for _ in manager.generate("no caption", voice):
            pass
    props.assert_awaited_once()


async def test_sidecar_failure_does_not_retry_on_torch(backend, monkeypatch):
    monkeypatch.setattr(backend, "_phonemes", Mock(return_value=iter(["həlˈOʊ"])))
    torch_generate = Mock()
    monkeypatch.setattr(KokoroV1, "generate", torch_generate)
    backend._sidecar.synthesize.side_effect = RuntimeError("sidecar unavailable")

    with pytest.raises(RuntimeError, match="sidecar unavailable"):
        async for _ in backend.generate("hello", ("af_heart", STOCK_PACK)):
            pass

    torch_generate.assert_not_called()
