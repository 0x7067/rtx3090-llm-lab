# nari-llama.cpp: Linux x86-64, CUDA 12.8

llama.cpp with the NariQ trellis tensor types (TRL1-TRL6) that the Qwen3.8-27B trellis GGUFs use. Stock llama.cpp,
Ollama and LM Studio cannot load those files.

- Create a destination folder, then extract this archive and the matching `cudart-nari-llama.cpp-...` archive with `tar -xzf ARCHIVE --strip-components=1 -C DESTINATION`. This places the programs and runtime libraries together.
- Needs:
  - an NVIDIA driver of 570 or newer;
  - glibc 2.34 or newer;
  - libstdc++ providing `GLIBCXX_3.4.30` (updated Ubuntu 22.04 or Debian 12 meet these runtime requirements);
  - libgomp.
- GPUs: real code for RTX 30xx, RTX 40xx / L4, H100 / H200 and RTX 50xx / RTX PRO 6000. Any other card of compute
  capability 8.0 or newer (A100, B200) compiles PTX on the first run, about 50 s.
- Without a GPU, the CPU backend runs, with the best variant for the processor chosen at start.

```bash
./llama-server -m Qwen3.8-27B-NariQ-3.0bpw.gguf --mmproj Qwen3.8-27B-NariQ-mmproj-BF16.gguf --port 8080
```

This opens a chat page and an OpenAI-compatible API at http://127.0.0.1:8080 (without `--port`, llama-server's
default is 9931). Without `-c`, the context is the longest
that free GPU memory holds, up to 262,144 tokens.

`build-info.json` gives the source revision and hashes. llama.cpp and the NariQ additions are under
the MIT license (`LICENSE`); libtrellis is Apache-2.0 (`LICENSE-libtrellis`); credits, including exllamav3, are in `NOTICE-NariQ` and `LICENSE-exllamav3`.
