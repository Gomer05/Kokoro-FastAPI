# Vulkan TTS (Linux x86-64)

Kokoro-FastAPI can send eligible speech requests to a separate llama.cpp Vulkan server. The same API endpoints, web player, and streaming formats are used. This is opt-in via `VULKAN_TTS_URL`; it does not change the default CPU, CUDA, or ROCm deployments. llama.cpp runs parts of Kokoro on the GPU and still uses the CPU for other stages.

- Vulkan uses unmodified stock voice packs included in the GGUF model. Custom `.pt` voices (including replacements for a stock voice), mixes, and requests for word timestamps use the existing Torch path. This includes WebSocket endpoints that need word timings. The deployment below runs Torch on CPU.
- The Vulkan and Torch engines can produce slightly different waveforms and durations for the same input.
- If the configured sidecar cannot serve an eligible request, the request fails. Kokoro does not silently retry it on CPU.
- The sidecar owns its model and GPU memory. Kokoro's `/dev/model` load state and unload controls describe the Torch model in the API process. Stop the sidecar to release its GPU memory.

## Docker Compose

Use a Linux x86-64 host with Docker Compose v2.24+ and a working GPU render node. The sidecar uses Mesa's Vulkan drivers and was verified on an AMD Radeon RX 580 (RADV). This setup does not cover proprietary NVIDIA drivers.

From the repository root, choose the render node for your GPU (see `ls -l /dev/dri/by-path`) and get the group ID from that node:

```bash
export VULKAN_RENDER_DEVICE=/dev/dri/renderD128
export VULKAN_RENDER_GID="$(stat -c '%g' "$VULKAN_RENDER_DEVICE")"
docker compose -f docker/vulkan/docker-compose.yml up --build
```

Compose builds the regular CPU API image from this checkout with `USE_GPU=false`. The sidecar runs as a non-root user with access to only the selected render node and its host group. It does not use `/dev/kfd` or privileged mode. The sidecar's HTTP port is not published; use Kokoro at `http://localhost:8880`.

If the sidecar does not become healthy, check `docker compose -f docker/vulkan/docker-compose.yml logs vulkan-tts`. If it cannot find `Vulkan0`, check the render node, its group and permissions, and the host kernel driver. Software Vulkan (Lavapipe) is not a fallback.

To check which GPU the sidecar sees and inspect its stock voices without publishing its port:

```bash
docker compose -f docker/vulkan/docker-compose.yml exec vulkan-tts \
  ./llama/llama-server --list-devices
docker compose -f docker/vulkan/docker-compose.yml exec vulkan-tts \
  curl -fsS http://127.0.0.1:8080/props
```

Use `docker compose -f docker/vulkan/docker-compose.yml down` to stop both services. To run CPU-only, start the regular CPU deployment without `VULKAN_TTS_URL` (including any value in the repository `.env`).

## Run the sidecar natively

Use a non-root user with read/write access to the GPU's render node, a Vulkan loader and hardware driver, and the runtime libraries for the Ubuntu x64 release (including glibc 2.34 or newer, `libvulkan.so.1`, `libstdc++.so.6`, `libgomp.so.1`, and OpenSSL 3). From the repository root:

```bash
repo="$PWD"
mkdir -p "$HOME/.local/share/kokoro-vulkan"
cd "$HOME/.local/share/kokoro-vulkan"
sh "$repo/docker/vulkan/download.sh"
ldd llama/llama-server llama/libggml-vulkan.so
env -u GGML_VK_VISIBLE_DEVICES 'VK_LOADER_DRIVERS_DISABLE=lvp*,*swiftshader*' \
  ./llama/llama-server --list-devices
```

The download script verifies SHA-256 checksums for the [llama.cpp b12494 release](https://github.com/ggml-org/llama.cpp/releases/tag/b12494) and both files from the [pinned Kokoro-82M-GGUF revision](https://huggingface.co/ggml-org/Kokoro-82M-GGUF/tree/ed4a98eab16351c7d803ff011e88267a5b036666). Resolve any `not found` lines from `ldd` before starting, and check that `--list-devices` lists your hardware as `Vulkan0`. If multiple GPUs are visible, you can use the Vulkan loader's `VK_DRIVER_FILES` to limit the drivers to the intended hardware ICD and check the device list again.

In that same directory, start the sidecar:

```bash
env -u GGML_VK_VISIBLE_DEVICES 'VK_LOADER_DRIVERS_DISABLE=lvp*,*swiftshader*' \
  ./llama/llama-server -m Kokoro-82M-F32.gguf -mm mmproj-Kokoro-82M-F16.gguf \
  --device Vulkan0 --mmproj-device Vulkan0 -ngl all \
  --host 127.0.0.1 --port 8080 --parallel 1 --no-ui
```

Keep `llama-server` on loopback or a private network. Its `GET /props` and `POST /tts` endpoints are for the Kokoro API, not public clients. With `uv` and `espeak-ng` installed, start the API from the repository root in another terminal:

```bash
curl -fsS http://127.0.0.1:8080/props
VULKAN_TTS_URL=http://127.0.0.1:8080 ./start-cpu.sh
```

Test either deployment through Kokoro, not the sidecar:

```bash
curl -fsS http://localhost:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","input":"Hello from Kokoro.","voice":"af_heart","response_format":"wav"}' \
  -o hello.wav
```

## Benchmark against PyTorch CPU

Use the same Linux host and Kokoro settings for both runs. Run the regular CPU
server first, then stop it and start the Vulkan deployment. With each server on
port 8880, run these from the repository root:

```bash
python examples/assorted_checks/benchmarks/benchmark_vulkan.py \
  --label pytorch-cpu --output /tmp/kokoro-cpu.json
# Stop the CPU server and start the Vulkan deployment before running again.
python examples/assorted_checks/benchmarks/benchmark_vulkan.py \
  --label vulkan --output /tmp/kokoro-vulkan.json
```

The script sends the same short and multi-chunk texts to `/v1/audio/speech`. It
reports time to first PCM sample, total latency, and real-time factor (wall time
divided by audio duration; lower is better). The label does not verify GPU use.
For a Vulkan run, also check that the sidecar processes synthesis during the
timed requests and that its GPU engine time increases (for example, the
`drm-engine-*` counters in `/proc/<llama-server-pid>/fdinfo/` on AMD Linux).

### Radeon RX 580 results

This native Linux comparison used a Ryzen 5 5600G, an 8 GB RX 580
(Mesa RADV 26.3.1), and one API worker. The baseline ran the unchanged
`5364fc3` checkout with PyTorch 2.8.0 CPU. The Vulkan run used the same
Kokoro frontend, `af_heart` voice, and PCM output through the API.
There were 20 timed requests per group and two separate warmup requests
per backend. AMD DRM counters confirmed that the sidecar submitted GPU work.

| Metric | PyTorch CPU p50 / p95 | Vulkan p50 / p95 |
| --- | ---: | ---: |
| Short: time to first PCM | 0.708 / 1.389 s | 0.374 / 0.618 s |
| Long: time to first PCM | 3.204 / 3.765 s | 1.261 / 1.296 s |
| Long: total time | 16.535 / 25.774 s | 6.580 / 10.443 s |
| Long: real-time factor | 0.229 / 0.243 | 0.093 / 0.095 |

Per-case median audio durations differed by less than 30 ms. These runs
measure single-client, warm performance on one machine, not cold starts
or concurrent requests.
