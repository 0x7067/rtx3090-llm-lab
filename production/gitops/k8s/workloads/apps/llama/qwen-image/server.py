#!/usr/bin/env python3
"""OpenAI images API for Qwen-Image 2.1, started by llama-swap on the 3090.

llama-swap launches this process when a request names a qwen-image model and
stops it when another model needs the card, so the GPU is never shared. Each
request runs up to two stages, one after the other, because they do not fit on
the 24 GB card together:

1. Prompt enhancer (Qwen-Image-2.1-PE-T2I for generation, PE-I2I for edits): a
   9B VLM that expands the brief and picks the canvas. It loads straight onto
   the GPU for the request and is freed before rendering. Skip it with
   ``"enhance": false``.
2. ``QwenImage21Pipeline``: the 8B text encoder, then the 7B DiT (~31 GB in bf16
   together) each load from disk straight onto the GPU and are freed after
   their step, so neither waits in host RAM.

Endpoints: ``GET /health``, ``POST /v1/images/generations`` (JSON) and
``POST /v1/images/edits`` (multipart, OpenAI shape). Each result carries
``b64_json`` plus the enhancer's ``revised_prompt`` and the seed used.
"""

import argparse
import base64
import gc
import inspect
import io
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Protocol

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pe_core as core  # noqa: E402

# Native 2K canvases from the Qwen-Image-2.1 README. Every edge is a multiple of
# 32, which the 16x VAE plus 2x patching needs.
RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}
NATIVE_AREA = 2048 * 2048
MIN_EDGE, MAX_EDGE = 256, 4096
MAX_EDIT_IMAGES = 4


class RequestError(ValueError):
    """A client error, reported as HTTP 400 in the OpenAI error shape."""


def snap32(value: float) -> int:
    return max(32, int(round(value / 32)) * 32)


def size_for_aspect(width: float, height: float) -> tuple[int, int]:
    """Native-area canvas for an aspect ratio, snapped to multiples of 32."""
    w = (NATIVE_AREA * width / height) ** 0.5
    return snap32(w), snap32(w * height / width)


def size_for_ratio(spec: str) -> tuple[int, int] | None:
    """Canvas for a ``W:H`` ratio: the README size when listed, else derived at
    the same area (the enhancer also emits 4:5, 21:9, ...). None if unparsable."""
    spec = (spec or "").strip()
    if spec in RATIOS:
        return RATIOS[spec]
    match = re.fullmatch(r"(\d+):(\d+)", spec)
    if not match or int(match.group(1)) == 0 or int(match.group(2)) == 0:
        return None
    return size_for_aspect(int(match.group(1)), int(match.group(2)))


def parse_size(size: Any) -> tuple[int, int] | None:
    """OpenAI ``size``: ``"auto"``/missing means "let the enhancer decide"."""
    if size in (None, "", "auto"):
        return None
    match = re.fullmatch(r"(\d+)x(\d+)", str(size).strip())
    if not match:
        raise RequestError(f"size {size!r} is not WIDTHxHEIGHT or auto")
    width, height = int(match.group(1)), int(match.group(2))
    for name, value in (("width", width), ("height", height)):
        if value % 32:
            raise RequestError(f"{name}={value} is not a multiple of 32; nearest is {snap32(value)}")
        if not MIN_EDGE <= value <= MAX_EDGE:
            raise RequestError(f"{name}={value} is outside {MIN_EDGE}..{MAX_EDGE}")
    return width, height


def choose_canvas(requested: tuple[int, int] | None, enhanced: dict[str, Any] | None,
                  image_sizes: list[tuple[int, int]]) -> tuple[int, int]:
    """Explicit size wins; then the enhancer's ratio_follow / wh_ratio; then the
    first source image's aspect; then the square native canvas."""
    if requested:
        return requested
    if enhanced:
        follow = re.fullmatch(r"<image(\d+)>", enhanced.get("ratio_follow") or "")
        if follow and 1 <= int(follow.group(1)) <= len(image_sizes):
            return size_for_aspect(*image_sizes[int(follow.group(1)) - 1])
        ratio = size_for_ratio(enhanced.get("wh_ratio") or "")
        if ratio:
            return ratio
    if image_sizes:
        return size_for_aspect(*image_sizes[0])
    return RATIOS["1:1"]


def fit_image(image: Any, max_pixels: int) -> Any:
    """Downscale so W*H <= max_pixels, as pe_core.load_image does for files."""
    width, height = image.size
    if max_pixels and width * height > max_pixels:
        scale = (max_pixels / float(width * height)) ** 0.5
        from PIL import Image
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
    return image


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
class Enhancer(Protocol):
    def enhance(self, task: str, prompt: str, images: list[Any], seed: int) -> dict[str, Any]: ...


class Renderer(Protocol):
    def render(self, prompt: str, width: int, height: int, seed: int,
               steps: int | None, images: list[Any]) -> Any: ...


class PromptEnhancer:
    """Qwen-Image-2.1-PE checkpoints, loaded per request and freed afterwards."""

    def __init__(self, checkpoints: dict[str, str]):
        self.checkpoints = checkpoints

    def enhance(self, task: str, prompt: str, images: list[Any], seed: int) -> dict[str, Any]:
        import torch
        from transformers import (AutoModelForImageTextToText, AutoProcessor,
                                  LogitsProcessor, LogitsProcessorList)

        ckpt = self.checkpoints.get(task)
        if not ckpt:
            raise RequestError(f"no prompt enhancer configured for {task}; send enhance=false")
        profile = core.get_profile(task)

        class PresencePenalty(LogitsProcessor):
            # vLLM-style presence penalty (subtract once per generated token id);
            # transformers has none, and t2i runs at 1.5.
            def __init__(self, penalty: float, prompt_len: int):
                self.penalty, self.prompt_len = penalty, prompt_len

            def __call__(self, input_ids, scores):
                for row in range(input_ids.shape[0]):
                    generated = input_ids[row, self.prompt_len:]
                    if generated.numel():
                        scores[row, generated.unique()] -= self.penalty
                return scores

        processor = AutoProcessor.from_pretrained(ckpt)
        # device_map streams shards onto the GPU: a host-RAM copy would sit on top
        # of the offloaded pipeline and push the pod past its memory limit.
        model = AutoModelForImageTextToText.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda").eval()
        try:
            fitted = [fit_image(im.convert("RGB"), profile.image_max_pixels) for im in images]
            messages = core.build_messages(core.load_system_prompt(None, ckpt), prompt, fitted)
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", enable_thinking=True).to(model.device)
            if "mm_token_type_ids" not in inputs and hasattr(processor, "create_mm_token_type_ids"):
                inputs["mm_token_type_ids"] = processor.create_mm_token_type_ids(inputs["input_ids"])
            prompt_len = inputs["input_ids"].shape[1]
            processors = LogitsProcessorList()
            if profile.presence_penalty:
                processors.append(PresencePenalty(profile.presence_penalty, prompt_len))
            torch.manual_seed(seed)
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=profile.max_new_tokens, do_sample=True,
                    temperature=profile.temperature, top_p=profile.top_p, top_k=profile.top_k,
                    logits_processor=processors, pad_token_id=processor.tokenizer.eos_token_id)
            text = processor.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
            _, answer = core.split_thinking(text)
            return core.parse_answer(answer, profile)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()


class QwenImageRenderer:
    """QwenImage21Pipeline with one large model on the GPU at a time, loaded from
    disk for its stage and freed after it, and tiled VAE decode.

    The text encoder and the DiT (~31 GB in bf16) do not fit the 24 GB card
    together. Model CPU offload would park both in host RAM, which the host
    cannot spare; reloading costs a read from the page cache or the NVMe instead.
    """

    STAGES = ("text_encoder", "transformer")

    def __init__(self, model_dir: str, default_steps: int | None):
        import torch
        from diffusers import QwenImage21Pipeline

        self.torch = torch
        self.model_dir = model_dir
        self.default_steps = default_steps
        self.pipe = QwenImage21Pipeline.from_pretrained(
            model_dir, text_encoder=None, transformer=None, torch_dtype=torch.bfloat16)
        self.pipe.vae.to("cuda")
        # Full-frame VAE decode at 2K needs one ~4.5 GiB activation; tiles fit.
        for name in ("enable_tiling", "enable_slicing"):
            if hasattr(self.pipe.vae, name):
                getattr(self.pipe.vae, name)()
        self.accepts = set(inspect.signature(self.pipe.__call__).parameters)
        self._wrap_encode_prompt()

    def _wrap_encode_prompt(self) -> None:
        """The pipeline encodes the prompt, then denoises: swap the models between."""
        encode_prompt = self.pipe.encode_prompt

        def staged_encode_prompt(*args: Any, **kwargs: Any):
            self._stage("text_encoder")
            encoded = encode_prompt(*args, **kwargs)
            self._stage("transformer")
            return encoded

        self.pipe.encode_prompt = staged_encode_prompt

    def _load(self, name: str) -> Any:
        """Read one model's shards straight onto the GPU, with no host-RAM copy."""
        if name == "text_encoder":
            from transformers import Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_dir, subfolder=name, dtype=self.torch.bfloat16, device_map="cuda").eval()
        from diffusers import QwenImage21Transformer2DModel
        return QwenImage21Transformer2DModel.from_pretrained(
            self.model_dir, subfolder=name, torch_dtype=self.torch.bfloat16, device_map="cuda").eval()

    def _drop(self, name: str) -> None:
        if getattr(self.pipe, name, None) is not None:
            self.pipe.register_modules(**{name: None})
            gc.collect()
            self.torch.cuda.empty_cache()

    def _stage(self, name: str) -> None:
        """Leave ``name`` as the only large model on the GPU."""
        for other in self.STAGES:
            if other != name:
                self._drop(other)
        if getattr(self.pipe, name, None) is None:
            self.pipe.register_modules(**{name: self._load(name)})

    def render(self, prompt: str, width: int, height: int, seed: int,
               steps: int | None, images: list[Any]) -> Any:
        kwargs: dict[str, Any] = {
            "prompt": prompt, "width": width, "height": height,
            "generator": self.torch.Generator("cpu").manual_seed(seed),
        }
        # Turbo carries its own 8-step sigma schedule; only override on request.
        if steps or self.default_steps:
            kwargs["num_inference_steps"] = steps or self.default_steps
        if "use_kv_cache" in self.accepts:
            kwargs["use_kv_cache"] = True
        if images:
            kwargs["image"] = images[0] if len(images) == 1 else images
        try:
            return self.pipe(**kwargs).images[0]
        finally:
            # Free the card for the next request's enhancer.
            for name in self.STAGES:
                self._drop(name)
            self.torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def read_options(body: dict[str, Any]) -> dict[str, Any]:
    """Validate the fields shared by generations and edits."""
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RequestError("prompt is required")
    if int(body.get("n") or 1) != 1:
        raise RequestError("n must be 1; send one request per image")
    if (body.get("response_format") or "b64_json") != "b64_json":
        raise RequestError("only response_format=b64_json is supported")
    steps = body.get("steps")
    if steps is not None:
        steps = int(steps)
        if not 1 <= steps <= 100:
            raise RequestError("steps must be 1..100")
    seed = body.get("seed")
    enhance = body.get("enhance")
    if enhance is None:  # absent from JSON, or a multipart field left out
        enhance = True
    if isinstance(enhance, str):
        enhance = enhance.strip().lower() not in ("0", "false", "no", "off")
    return {
        "prompt": prompt.strip(),
        "size": parse_size(body.get("size")),
        "steps": steps,
        "seed": int(seed) if seed is not None else random.randrange(2**31),
        "enhance": bool(enhance),
    }


def run_request(task: str, options: dict[str, Any], images: list[Any],
                enhancer: Enhancer, renderer: Renderer, served_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    enhanced = None
    prompt = options["prompt"]
    if options["enhance"]:
        enhanced = enhancer.enhance(task, prompt, images, options["seed"])
        prompt = enhanced.get("positive_prompt") or prompt
    enhanced_at = time.perf_counter()
    width, height = choose_canvas(options["size"], enhanced, [im.size for im in images])
    image = renderer.render(prompt, width, height, options["seed"], options["steps"], images)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    finished = time.perf_counter()
    return {
        "created": int(time.time()),
        "model": served_name,
        "output_format": "png",
        "size": f"{width}x{height}",
        "data": [{
            "b64_json": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "revised_prompt": prompt,
            "seed": options["seed"],
        }],
        "timings": {
            "enhance_s": round(enhanced_at - started, 1),
            "render_s": round(finished - enhanced_at, 1),
            "enhance_parse_ok": enhanced.get("parse_ok") if enhanced else None,
        },
    }


def build_app(enhancer: Enhancer, renderer: Renderer | None, served_name: str):
    from fastapi import FastAPI, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse
    from PIL import Image

    app = FastAPI(title="qwen-image")
    # One GPU: requests queue here rather than racing for VRAM.
    gpu = threading.Lock()

    def error(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
        return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)

    def locked(task: str, options: dict[str, Any], images: list[Any]) -> dict[str, Any]:
        with gpu:
            return run_request(task, options, images, enhancer, renderer, served_name)

    async def respond(task: str, options: dict[str, Any], images: list[Any]):
        try:
            return await run_in_threadpool(locked, task, options, images)
        except RequestError as exc:
            return error(400, str(exc))
        except Exception as exc:  # surfaced to the caller; the process stays up
            return error(500, f"{type(exc).__name__}: {exc}", "server_error")

    @app.get("/health")
    def health():
        return {"status": "ok"} if renderer is not None else error(503, "loading", "unavailable")

    @app.post("/v1/images/generations")
    async def generations(request: Request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise RequestError("body must be a JSON object")
            options = read_options(body)
        except RequestError as exc:
            return error(400, str(exc))
        except ValueError as exc:
            return error(400, f"invalid request: {exc}")
        return await respond("t2i", options, [])

    @app.post("/v1/images/edits")
    async def edits(request: Request):
        try:
            form = await request.form()
            if form.get("mask") is not None:
                raise RequestError("mask is not supported; describe the change in the prompt")
            uploads = [*form.getlist("image"), *form.getlist("image[]")]
            if not uploads:
                raise RequestError("at least one image is required")
            if len(uploads) > MAX_EDIT_IMAGES:
                raise RequestError(f"at most {MAX_EDIT_IMAGES} images")
            images = []
            for upload in uploads:
                data = await upload.read() if hasattr(upload, "read") else None
                if not data:
                    raise RequestError("image fields must be file uploads")
                try:
                    images.append(Image.open(io.BytesIO(data)).convert("RGBA"))
                except Exception as exc:
                    raise RequestError(f"could not read image: {exc}") from exc
            options = read_options({key: form.get(key) for key in
                                    ("prompt", "n", "response_format", "size", "steps", "seed", "enhance")})
        except RequestError as exc:
            return error(400, str(exc))
        except ValueError as exc:
            return error(400, f"invalid request: {exc}")
        return await respond("edit", options, images)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--model", required=True, help="QwenImage21Pipeline checkpoint directory")
    parser.add_argument("--pe-t2i", required=True)
    parser.add_argument("--pe-i2i", required=True)
    parser.add_argument("--default-steps", type=int, default=None,
                        help="Steps when the request sets none (Turbo: leave unset for its schedule)")
    args = parser.parse_args()

    import uvicorn

    print(f"loading {args.model}", flush=True)
    started = time.perf_counter()
    renderer = QwenImageRenderer(args.model, args.default_steps)
    print(f"pipeline ready in {time.perf_counter() - started:.0f}s", flush=True)
    enhancer = PromptEnhancer({"t2i": args.pe_t2i, "edit": args.pe_i2i})
    uvicorn.run(build_app(enhancer, renderer, args.served_name), host=args.host, port=args.port,
                log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
