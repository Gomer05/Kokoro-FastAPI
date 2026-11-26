#!/usr/bin/env python3
"""Compare single-request streaming performance via POST /v1/audio/speech.

Run each backend in isolation on the same host, using the same benchmark options:

    python benchmark_vulkan.py --output cpu.json
    python benchmark_vulkan.py --base-url http://127.0.0.1:8881 --label vulkan \
        --output vulkan.json

Latency is measured at the client, from submitting the request to reading the
first complete PCM sample or the end of the response. RTF is wall time divided
by audio duration (lower is better). The label does not verify the backend.
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen

SPEECH_PATH = "/v1/audio/speech"
SAMPLE_RATE = 24_000
BYTES_PER_SAMPLE = 2
SPEECH_OPTIONS = {
    "model": "kokoro",
    "voice": "af_heart",
    "speed": 1.0,
    "response_format": "pcm",
    "stream": True,
}

SHORT = (
    "The kettle is ready.",
    "Please leave the spare key under the blue flowerpot.",
    "Could you tell me whether the next train stops at Oak Street?",
    "At 7:45 tomorrow morning, the library will open its doors for the first time since Monday.",
)

HARBOR = (
    "The small harbor looked quiet at dawn. A line of fishing boats rocked against the "
    "wooden pier, and a pale light moved slowly across the water. By the time the bakery "
    "opened, three neighbors had gathered outside to talk about the weather. One of them "
    "had brought a paper bag full of plums from the tree behind her house. She passed "
    "them around without counting, then put the last one back in the bag for the baker."
)
TRAIL = (
    "We followed the trail uphill until the road disappeared behind a stand of pine trees. "
    "The first mile was easy, but the rocks grew loose near the ridge, and loose stones "
    "kept slipping from under our boots. At the highest point we stopped to share an apple "
    "and study the map. There were two paths into the valley. The longer one curved gently "
    "through a meadow, while the shorter one ran past an old mill and crossed a narrow "
    "bridge. We chose the bridge, knowing full well that we might have to turn back."
)
STATION = (
    "The station clock read ten past nine, but the ticket clerk said the southbound train "
    "would not leave until half past. Mina carried her suitcase into the waiting room and "
    "sat beside a window. The glass was cool against her hand. Across the tracks, a man in "
    "a green coat was trying to fold a newspaper in the wind. Every time he got one corner "
    "into place, another corner slipped away. She watched him try again, then laughed when "
    "he finally tucked the whole bundle under his arm and went in search of coffee."
)
GARDEN = (
    "After the rain, white blossoms lay scattered across the garden path. Their edges had "
    "begun to turn brown, but the air still smelled of jasmine and wet earth. Near the gate, "
    "a snail had climbed halfway up a watering can. Nobody had walked that way for a week, "
    "so a small branch blocked the path. I moved it aside before turning to the raised beds, "
    "where the peas had already sent thin green shoots through the straw. If the sun came "
    "out before lunch, I would build them a simple frame to climb."
)
LONG = (
    HARBOR,
    f"{TRAIL}\n\n{GARDEN}",
    f"{STATION}\n\n{HARBOR}\n\n{TRAIL}",
    f"{GARDEN}\n\n{TRAIL}\n\n{HARBOR}\n\n{STATION}",
)
CASES = {
    f"{group}-{idx}": text
    for group, texts in (("short", SHORT), ("long", LONG))
    for idx, text in enumerate(texts, start=1)
}
WARMUP = {
    "warmup-short": "We will begin as soon as everyone has found a seat.",
    "warmup-long": f"{TRAIL}\n\n{STATION}",
}
METRICS = ("first_pcm_seconds", "total_seconds", "rtf", "audio_seconds")


def measure(url: str, text: str, timeout: float) -> dict:
    request = Request(
        url,
        data=json.dumps({**SPEECH_OPTIONS, "input": text}).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
            "Accept-Encoding": "identity",
        },
        method="POST",
    )
    start = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"expected HTTP 200, got {response.status}")
        media_type = response.headers.get_content_type()
        if media_type != "audio/pcm":
            raise RuntimeError(f"expected audio/pcm, got {media_type}")

        first_sample = response.read(BYTES_PER_SAMPLE)
        first_pcm_seconds = time.perf_counter() - start
        if len(first_sample) != BYTES_PER_SAMPLE:
            raise RuntimeError("response did not contain a complete PCM sample")

        pcm_bytes = len(first_sample)
        non_silent = any(first_sample)
        while chunk := response.read(64 * 1024):
            pcm_bytes += len(chunk)
            if not non_silent:
                non_silent = bool(chunk.strip(b"\x00"))
        total_seconds = time.perf_counter() - start

    if pcm_bytes % BYTES_PER_SAMPLE:
        raise RuntimeError(f"PCM stream ended mid-sample ({pcm_bytes} bytes)")
    if not non_silent:
        raise RuntimeError("PCM stream contained only zero samples")
    audio_seconds = pcm_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    # Catch empty fallbacks and grossly truncated or misformatted streams.
    words = len(text.split())
    if not max(0.5, words / 10) <= audio_seconds <= max(4, words):
        raise RuntimeError(
            f"implausible PCM duration: {audio_seconds:.3f}s for {words} words"
        )
    return {
        "first_pcm_seconds": first_pcm_seconds,
        "total_seconds": total_seconds,
        "pcm_bytes": pcm_bytes,
        "audio_seconds": audio_seconds,
        "rtf": total_seconds / audio_seconds,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (index - low) * (ordered[high] - ordered[low])


def summarize(samples: list[dict]) -> dict:
    return {
        "n": len(samples),
        **{
            metric: {
                "p50": percentile([sample[metric] for sample in samples], 0.50),
                "p95": percentile([sample[metric] for sample in samples], 0.95),
            }
            for metric in METRICS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8880")
    parser.add_argument("--label", default="cpu", help="backend label in the results")
    parser.add_argument("--rounds", type=int, default=5, help="timed runs per case")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="passes over the two warmup cases (0 skips)",
    )
    parser.add_argument(
        "--timeout", type=float, default=120, help="socket timeout in seconds"
    )
    parser.add_argument(
        "--output", type=Path, help="save raw samples and summary as JSON"
    )
    args = parser.parse_args()
    if (
        args.rounds < 1
        or args.warmup < 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error(
            "--rounds and --timeout must be positive; --warmup must be nonnegative"
        )

    url = args.base_url.rstrip("/") + SPEECH_PATH
    started_utc = datetime.now(timezone.utc).isoformat()
    print(f"{args.label}: {url} ({args.rounds} rounds, {args.warmup} warmup passes)")

    def run(case: str, text: str, round_number: int, phase: str) -> dict:
        try:
            result = measure(url, text, args.timeout)
        except (OSError, HTTPException, RuntimeError, ValueError) as exc:
            parser.exit(1, f"{phase} {round_number}, {case}: {exc}\n")
        print(
            f"{phase:>6} {round_number:2d} {case:<13} "
            f"first={result['first_pcm_seconds']:.3f}s  total={result['total_seconds']:.3f}s  "
            f"audio={result['audio_seconds']:.3f}s  RTF={result['rtf']:.4f}",
            flush=True,
        )
        return {"round": round_number, "case": case, **result}

    warmup = [
        run(case, text, round_number, "warmup")
        for round_number in range(1, args.warmup + 1)
        for case, text in WARMUP.items()
    ]
    samples = []
    for round_number in range(1, args.rounds + 1):
        for offset in range(len(SHORT)):
            idx = (round_number - 1 + offset) % len(SHORT) + 1
            for group in ("short", "long"):
                case = f"{group}-{idx}"
                samples.append(run(case, CASES[case], round_number, "timed"))

    summary = {
        group: summarize(
            [sample for sample in samples if sample["case"].startswith(group)]
        )
        for group in ("short", "long")
    }
    print("\np50 / p95 (linearly interpolated)")
    for group, stats in summary.items():
        print(f"{group} (n={stats['n']})")
        for metric in METRICS:
            values = stats[metric]
            print(f"  {metric:>17}: {values['p50']:.4f} / {values['p95']:.4f}")

    if args.output:
        report = {
            "label": args.label,
            "url": url,
            "started_utc": started_utc,
            "request_options": SPEECH_OPTIONS,
            "pcm": {"sample_rate": SAMPLE_RATE, "channels": 1, "format": "s16le"},
            "rounds": args.rounds,
            "warmup_passes": args.warmup,
            "socket_timeout_seconds": args.timeout,
            "cases": CASES,
            "warmup_cases": WARMUP,
            "warmup_samples": warmup,
            "samples": samples,
            "summary": summary,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
