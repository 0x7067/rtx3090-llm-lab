# Private OCR deployment artifacts

These files implement the selected CPU layout + GPU vLLM deployment path. They have **not passed an image build or real OCR run**. The local Docker daemon is unavailable; no image layer, model weight, or package installation was performed on the home server. The example manifest intentionally fails startup because final built-image digests, adapter source snapshot, and downloaded model inventories do not exist yet.

The selected server has an RTX 3090 (compute 8.6), driver 595.91.07, NVIDIA Container Toolkit 1.20.1, Docker 29.9.1 and Compose 5.6.0. An existing `VLLM::EngineCore` workload occupies about 23.2GiB; roughly 839MiB was free on the latest inspection. This profile's **10GiB free-memory check is a declared trial budget, not a measured minimum**. Existing workloads must not be stopped or repurposed without the user's sharing decision.

## What runs

- `ocr-api`: Node 24.11.0, Python 3.12.15, PaddleOCR 3.7.0/PaddleX 3.7.2, CPU Paddle 3.2.1, Pillow 11.3.0, PDFium wrapper 4.30.0. The Node service owns the durable queue and starts one killable Python document worker. CPU layout reserves GPU memory for visual-language recognition.
- `ocr-vllm`: Python 3.12.15, PaddleX 3.7.2's registered model plugin, vLLM 0.10.2, Torch 2.8.0 and Transformers 4.55.4. Both environments pin the OpenAI-compatible client SDK to 1.109.1 and connect only to the private vLLM endpoint. The locked wheels supply CUDA 12.8 user-space libraries, independently of the host Python and CUDA toolkit. This is a custom Python-base image, not the stock vLLM image.
- Both services require the `ocr-real` profile and `OCR_ENABLED=real`. Only `127.0.0.1:9417` is published. The model endpoint and metrics remain on an internal Docker network without runtime internet access. Model directories are read-only; no Docker socket is mounted. There must be no other client of this dedicated model service.
- vLLM permits one sequence and one API server, with an explicit 0.35 GPU memory allocation fraction and 8192-token context/batch limits. The API also waits for dedicated vLLM running/waiting counters to reach zero after Python exits before releasing its document slot. An unreachable/busy model endpoint holds the slot instead of submitting more work.

The upstream processor requires `trust-remote-code`. Here that executes only the staged, pinned, SHA-256-inventoried model repository code with offline mode and no runtime internet. Review those model code files before first enablement. Package/model pins are supported by the [runtime evidence](../../../docs/ocr/RUNTIME.md); upstream [PaddleX CLI](https://github.com/PaddlePaddle/PaddleX/blob/v3.7.2/paddlex/inference/genai/server.py) and [1.6 backend configuration](https://github.com/PaddlePaddle/PaddleX/blob/v3.7.2/paddlex/inference/genai/configs/paddleocr_vl_16_09b.py) define the actual invocation.

## Dependency and build checks

Both complete transitive locks were resolved with `uv 0.6.10`, Python 3.12, Linux x86_64/glibc 2.28, binary-only artifacts and a 23 October 2026 UTC cutoff. The layout lock has 101 packages; vLLM has 176. Hashes come from registry metadata; `pip --require-hashes --only-binary=:all:` verifies downloaded wheels during the later build. Resolver success does not prove import or CUDA compatibility. The Dockerfiles run `pip check`, check every installed locked version, and retain OS/Python/Node inventories in `/opt/ocr-build/`.

Regenerate intentionally, review the diff, then update the example manifest's lock hashes:

```sh
uv pip compile services/ocr/deploy/requirements.layout.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 --no-python-downloads --only-binary :all: --generate-hashes --exclude-newer 2026-10-23T00:00:00Z -o services/ocr/deploy/requirements.layout.lock
uv pip compile services/ocr/deploy/requirements.vllm.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 --no-python-downloads --only-binary :all: --generate-hashes --exclude-newer 2026-10-23T00:00:00Z -o services/ocr/deploy/requirements.vllm.lock
python3 -B -m unittest discover -s services/ocr/deploy -p 'test_*.py' -v
```

After choosing an authorized build machine, from the repository root:

```sh
OCR_ADAPTER_TREE_SHA256=$(python3 services/ocr/deploy/source_inventory.py)
docker build --platform linux/amd64 --build-arg OCR_ADAPTER_TREE_SHA256="$OCR_ADAPTER_TREE_SHA256" -f services/ocr/deploy/Dockerfile.layout -t finance-ocr-layout:candidate .
docker build --platform linux/amd64 --build-arg OCR_ADAPTER_TREE_SHA256="$OCR_ADAPTER_TREE_SHA256" -f services/ocr/deploy/Dockerfile.vllm -t finance-ocr-vllm:candidate .
docker compose -f services/ocr/deploy/compose.yaml config --quiet
docker compose -f services/ocr/deploy/compose.yaml --profile ocr-real config --quiet
```

These commands build/configure; they do not start OCR. Dockerfile-specific ignore files limit the context to required source and exclude `deploy/local/`, so source documents, credentials and model files are not baked into images. The base Python and Node images are pinned to linux/amd64 registry digests. A dated Debian snapshot fixes the apt repository; its availability and the full resulting OS inventory remain build checks. This is not a security audit or final SBOM.

## Prepare the reviewed runtime

1. Snapshot the reviewed adapter using `source_inventory.py`. Its hash covers the exact copied sources and package locks; both builds reject a different snapshot and label the resulting image with that hash. Record actual immutable image IDs (`sha256:...`) from `docker image inspect`, or repository digests if using an explicitly authorized private registry. Local IDs allow a build on the target without publishing anything. Base-image digests are not final-image identifiers. Record the source hash in `artifacts.adapterTreeSha256`; container preflight recomputes it and compares its build inventory. `check_launch.py` separately verifies local Docker image IDs/repository digests, platform and source labels before startup.
2. Once model downloads are authorized, stage the exact OCR/layout revisions from `manifest.example.json` into separate `models/ocr` and `models/layout` directories. Include configuration, tokenizer and processor code; retain license notices. Use ordinary files, without symlinks or a Hugging Face cache tree. Confirm revision provenance from the download record, inspect model code, then run `python3 services/ocr/deploy/inventory_models.py --models-dir /absolute/staged/models`. This command only inventories local bytes; it does not establish that arbitrary local files came from the declared revision.
3. Create a private `services/ocr/deploy/local/` directory (mode `0700`). Copy `manifest.example.json` to `local/manifest.json`, fill the actual image references and adapter source hash, and replace `models` with the inventory output. The complete manifest is canonicalized into the pipeline fingerprint; lock hashes, model hashes, inference/decoder settings and normalization revision all participate. Never replace absent evidence with dummy hashes.
4. Create a random service token of at least 24 characters in a private file readable by container UID 10001. Compose file-backed secrets retain host permissions; verify readability without making the token world-readable. Set `OCR_TOKEN_PATH`, `OCR_NAMESPACE`, `OCR_MANIFEST_PATH`, `OCR_MODELS_DIR`, `OCR_LAYOUT_IMAGE`, and `OCR_VLLM_IMAGE` to the reviewed values. Use absolute paths. No default credential exists. Do not put tokens in the manifest or repository.
5. Resolve GPU sharing/capacity, then inspect the configured profile again. Run `python3 services/ocr/deploy/check_launch.py --manifest /absolute/manifest.json --models-dir /absolute/models --token-file /absolute/service-token --namespace <namespace>` to verify staging, image identity and Compose configuration without starting services. Only when an actual synthetic run is authorized, repeat that command with `--start`. The launcher sets the explicit real-mode gate and invokes Compose with `--no-build --pull never`. Both containers fail closed on an incomplete manifest, wrong versions or missing/changed model files. vLLM also checks free GPU memory before loading. This check does not reserve memory against unrelated processes; coordinate the GPU allocation before launch.

The host Python version is irrelevant to inference. The app's current client allows literal loopback HTTP only. Keep the app on the same host or use an authenticated SSH tunnel such as `ssh -N -L 9417:127.0.0.1:9417 denguinho@192.168.3.34`; this is a future operator command, not an action performed by this task. Direct LAN access requires a separately reviewed authenticated TLS configuration. No real financial document should be transmitted until the destination and evaluation conditions are met.

## Build-only transfer proposal

No transfer or remote build has been performed. After authorization, package the reviewed source with `python3 services/ocr/deploy/source_inventory.py --archive /tmp/finance-ocr-source.tar.gz`. The archive uses the same source allowlist as both images; it omits credentials, app databases, node_modules, `.git`, local models and document data. Transfer it to a new dedicated directory on the selected server and run only the two build commands above. Building downloads image layers/locked packages and consumes host disk/RAM/CPU, but neither command starts a GPU process or stops existing services. Inspect build inventories and retain image IDs before proposing model staging or a synthetic run.

## Remaining proof

Complete image builds, verify imports and model registration, record final image/SBOM/license inventory, and run synthetic receipts with measured GPU/host memory and evidence-coordinate checks. Exercise decode failures, PDF rotation, EXIF, process cancellation and vLLM drain failures under the actual containers. Retain the existing manual-entry workflow. Passing local preflight tests proves configuration rejection behavior only; it does not prove OCR quality or deployment readiness.
