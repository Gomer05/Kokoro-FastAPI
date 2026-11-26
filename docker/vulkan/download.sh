#!/bin/sh
set -eu

download() {
    curl --fail --silent --show-error --location --retry 3 \
        --proto '=https' --proto-redir '=https' --output "$2" "$1"
    printf '%s  %s\n' "$3" "$2" | sha256sum --check --strict
}

release=b12494
archive="llama-${release}-bin-ubuntu-vulkan-x64.tar.gz"
download "https://github.com/ggml-org/llama.cpp/releases/download/${release}/${archive}" \
    "$archive" 0dbf265d5742cbdd43380f98fee6a3a4352746d0818cd2a700fe332bcb312e7d

models=https://huggingface.co/ggml-org/Kokoro-82M-GGUF/resolve/ed4a98eab16351c7d803ff011e88267a5b036666
download "$models/Kokoro-82M-F32.gguf?download=true" \
    Kokoro-82M-F32.gguf be19298184f1a2577eb20e4689f1e2b315cbf474c39e113f192991fef3c85c6e
download "$models/mmproj-Kokoro-82M-F16.gguf?download=true" \
    mmproj-Kokoro-82M-F16.gguf 071d20f53b7c5d64b08a24bff5dfb77c2ea0df843a7c6c2356998a735ec6929d

mkdir -p llama
tar -xzf "$archive" --strip-components=1 --no-same-owner -C llama
rm "$archive"
