# syntax=docker/dockerfile:1
FROM debian:trixie-slim

RUN test "$(dpkg --print-architecture)" = amd64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl libbrotli1 libgomp1 libssl3t64 libstdc++6 \
        libvulkan1 libzstd1 mesa-vulkan-drivers zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --user-group --uid 10001 tts

WORKDIR /opt/tts
COPY download.sh /usr/local/bin/download-vulkan-tts
COPY NOTICE ./
RUN sh /usr/local/bin/download-vulkan-tts \
    && cp /usr/share/common-licenses/Apache-2.0 ./ \
    && ldd llama/llama-server llama/libggml-*.so > /tmp/llama-ldd \
    && cat /tmp/llama-ldd \
    && ! grep -q 'not found' /tmp/llama-ldd \
    && rm /tmp/llama-ldd

ENV VK_LOADER_DRIVERS_DISABLE=lvp*
USER tts
RUN ./llama/llama-server --version

ENTRYPOINT ["./llama/llama-server", "-m", "Kokoro-82M-F32.gguf", "-mm", "mmproj-Kokoro-82M-F16.gguf", "--device", "Vulkan0", "--mmproj-device", "Vulkan0", "-ngl", "all"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--parallel", "1", "--no-ui"]
