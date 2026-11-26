import asyncio
import json
import struct

import httpx
import numpy as np
import pytest

from api.src.inference import llama_tts
from api.src.inference.llama_tts import LlamaTTS, LlamaTTSError

PCM_CONTENT_TYPE = "audio/pcm;rate=24000;encoding=float;bits=32"


@pytest.fixture
def sidecar(monkeypatch):
    async_client = httpx.AsyncClient

    def make(handler):
        def client(**kwargs):
            assert kwargs["follow_redirects"] is False
            assert kwargs["trust_env"] is False
            return async_client(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", client)
        return LlamaTTS(base_url="http://sidecar.local:8080/kokoro/")

    return make


async def test_voices_reads_b12494_props(sidecar):
    def handler(request):
        assert request.method == "GET"
        assert str(request.url) == "http://sidecar.local:8080/kokoro/props"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "model_alias": "Kokoro-82M-F32.gguf",
                "modalities": {"vision": False, "video": False, "audio": False},
                "voices": ["af_heart", "am_adam", "af_heart"],
            },
        )

    voices = await sidecar(handler).voices()
    assert isinstance(voices, frozenset)
    assert voices == {"af_heart", "am_adam"}


@pytest.mark.parametrize(
    "props",
    [
        {"modalities": {"audio": True}},
        {"voices": []},
        {"voices": ["  "]},
        {"voices": ["af_heart", 3]},
        {"voices": {"af_heart": "American English"}},
        ["af_heart"],
    ],
)
async def test_voices_requires_valid_advertised_voices(sidecar, props):
    client = sidecar(lambda _: httpx.Response(200, json=props))
    with pytest.raises(LlamaTTSError, match="GET /props did not report valid voices"):
        await client.voices()


async def test_voices_rejects_invalid_json(sidecar):
    client = sidecar(lambda _: httpx.Response(200, content=b"<html>"))
    with pytest.raises(LlamaTTSError, match="GET /props did not return JSON"):
        await client.voices()


async def test_synthesize_sends_raw_prompt_and_returns_owned_audio(sidecar):
    payload = struct.pack("<3f", -0.25, 0.5, 0.125)

    def handler(request):
        assert request.method == "POST"
        assert str(request.url) == "http://sidecar.local:8080/kokoro/tts"
        assert "authorization" not in request.headers
        assert json.loads(request.content) == {
            "input": "həlˈOʊ wˈɜɹld.",
            "voice": "af_heart",
            "speed": 1.25,
            "raw_prompt": True,
            "response_format": "pcm",
            "stream": False,
        }
        return httpx.Response(
            200, content=payload, headers={"Content-Type": PCM_CONTENT_TYPE}
        )

    client = sidecar(handler)
    first = await client.synthesize("həlˈOʊ wˈɜɹld.", "af_heart", 1.25)
    second = await client.synthesize("həlˈOʊ wˈɜɹld.", "af_heart", 1.25)
    assert first.dtype == np.float32
    assert first.ndim == 1
    assert first.flags.writeable and first.flags.owndata
    np.testing.assert_array_equal(first, [-0.25, 0.5, 0.125])
    first[0] = 0
    assert second[0] == -0.25


@pytest.mark.parametrize(
    ("content_type", "payload", "message"),
    [
        ("application/json", struct.pack("<f", 0.1), "mono 24 kHz float32 PCM"),
        (
            PCM_CONTENT_TYPE.replace("24000", "48000"),
            struct.pack("<f", 0.1),
            "mono 24 kHz float32 PCM",
        ),
        (
            PCM_CONTENT_TYPE + ";channels=2",
            struct.pack("<2f", 0.1, 0.2),
            "mono 24 kHz float32 PCM",
        ),
        (
            "audio/pcm;rate=24000;encoding=signed-int;bits=16",
            b"\x00\x01\x00\x01",
            "mono 24 kHz float32 PCM",
        ),
        (PCM_CONTENT_TYPE, b"", "empty or truncated PCM"),
        (PCM_CONTENT_TYPE, b"\x00", "empty or truncated PCM"),
        (PCM_CONTENT_TYPE, struct.pack("<f", float("nan")), "non-finite PCM"),
        (PCM_CONTENT_TYPE, struct.pack("<f", float("inf")), "non-finite PCM"),
    ],
)
async def test_synthesize_rejects_invalid_audio(
    sidecar, content_type, payload, message
):
    client = sidecar(
        lambda _: httpx.Response(
            200, content=payload, headers={"Content-Type": content_type}
        )
    )
    with pytest.raises(LlamaTTSError, match=message):
        await client.synthesize("həlˈOʊ.", "af_heart", 1.0)


@pytest.mark.parametrize("speed", [0.0, -1.0, float("nan"), float("inf")])
async def test_invalid_speed_never_requests_audio(sidecar, speed):
    client = sidecar(lambda _: pytest.fail("Invalid speed reached the sidecar"))
    with pytest.raises(LlamaTTSError, match="finite positive speed"):
        await client.synthesize("həlˈOʊ.", "af_heart", speed)


@pytest.mark.parametrize(
    "url",
    [
        "/sidecar",
        "ftp://sidecar.local",
        "http://operator:secret@sidecar.local",
        "http://sidecar.local?secret=1",
        "http://sidecar.local#secret",
        "http://sidecar.local:secret",
    ],
)
def test_invalid_base_url_does_not_expose_credentials(url):
    with pytest.raises(LlamaTTSError, match="HTTP\\(S\\) base URL") as error:
        LlamaTTS(base_url=url)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("status", [307, 400, 503])
async def test_http_errors_do_not_forward_or_expose_request_data(sidecar, status):
    def handler(request):
        assert request.url.host == "sidecar.local"
        return httpx.Response(
            status,
            headers={"Location": "https://other.local/"},
            text="sidecar error echoing private speech",
        )

    with pytest.raises(
        LlamaTTSError, match=f"POST /tts returned HTTP {status}"
    ) as error:
        await sidecar(handler).synthesize("private speech", "af_heart", 1.0)
    assert "private speech" not in str(error.value)


async def test_connection_error_does_not_expose_transport_details(sidecar):
    def handler(request):
        raise httpx.ConnectError("private connection details", request=request)

    with pytest.raises(LlamaTTSError, match="GET /props failed") as error:
        await sidecar(handler).voices()
    assert "private connection details" not in str(error.value)


async def test_caps_response_size(sidecar, monkeypatch):
    monkeypatch.setattr(llama_tts, "_MAX_PROPS_BYTES", 4)
    client = sidecar(lambda _: httpx.Response(200, json={"voices": ["af_heart"]}))
    with pytest.raises(LlamaTTSError, match="unexpectedly large response"):
        await client.voices()


async def test_total_request_deadline(sidecar, monkeypatch):
    monkeypatch.setattr(llama_tts, "_PROPS_TIMEOUT", 0.01)

    async def handler(_):
        await asyncio.Future()

    with pytest.raises(LlamaTTSError, match="GET /props timed out"):
        await sidecar(handler).voices()


async def test_cancellation_closes_response(sidecar):
    reading = asyncio.Event()
    closed = asyncio.Event()

    class PendingAudio(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield struct.pack("<f", 0.1)
            reading.set()
            await asyncio.Future()

        async def aclose(self):
            closed.set()

    client = sidecar(
        lambda _: httpx.Response(
            200, stream=PendingAudio(), headers={"Content-Type": PCM_CONTENT_TYPE}
        )
    )
    task = asyncio.create_task(client.synthesize("həlˈOʊ.", "af_heart", 1.0))
    await asyncio.wait_for(reading.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
