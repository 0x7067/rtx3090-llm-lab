"""Request handling without a GPU: the enhancer and renderer are fakes."""

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server


class FakeEnhancer:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"positive_prompt": "an expanded prompt", "wh_ratio": "16:9",
                                 "ratio_follow": "", "parse_ok": True}

    def enhance(self, task, prompt, images, seed):
        self.calls.append((task, prompt, len(images), seed))
        return dict(self.result)


class FakeRenderer:
    def __init__(self):
        self.calls = []

    def render(self, prompt, width, height, seed, steps, images):
        self.calls.append({"prompt": prompt, "size": (width, height), "seed": seed,
                           "steps": steps, "images": len(images)})
        return Image.new("RGB", (8, 8), "red")


def client(enhancer=None, renderer=None):
    enhancer = enhancer or FakeEnhancer()
    renderer = renderer or FakeRenderer()
    return TestClient(server.build_app(enhancer, renderer, "qwen-image-test")), enhancer, renderer


def png_bytes(size=(300, 200)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def test_ratios_and_sizes():
    assert server.size_for_ratio("16:9") == (2752, 1536)
    width, height = server.size_for_ratio("4:5")
    assert width % 32 == 0 and height % 32 == 0 and width < height
    assert server.size_for_ratio("nonsense") is None
    assert server.parse_size("auto") is None
    assert server.parse_size("1024x768") == (1024, 768)
    with pytest.raises(server.RequestError):
        server.parse_size("1000x768")
    with pytest.raises(server.RequestError):
        server.parse_size("8192x8192")


def test_canvas_precedence():
    assert server.choose_canvas((512, 512), {"wh_ratio": "16:9"}, []) == (512, 512)
    follow = server.choose_canvas(None, {"ratio_follow": "<image2>", "wh_ratio": ""}, [(100, 100), (300, 100)])
    assert follow[0] > follow[1]
    assert server.choose_canvas(None, {"wh_ratio": "9:16", "ratio_follow": ""}, []) == (1536, 2752)
    assert server.choose_canvas(None, None, []) == (2048, 2048)


def test_generation_enhances_then_renders_at_the_enhancer_ratio():
    http, enhancer, renderer = client()
    response = http.post("/v1/images/generations", json={"prompt": "a corgi", "seed": 7})
    assert response.status_code == 200, response.text
    body = response.json()
    assert enhancer.calls == [("t2i", "a corgi", 0, 7)]
    assert renderer.calls[0]["prompt"] == "an expanded prompt"
    assert renderer.calls[0]["size"] == (2752, 1536)
    assert body["size"] == "2752x1536" and body["model"] == "qwen-image-test"
    assert body["data"][0]["revised_prompt"] == "an expanded prompt"
    assert body["data"][0]["seed"] == 7
    assert base64.b64decode(body["data"][0]["b64_json"]).startswith(b"\x89PNG")


def test_generation_without_enhancer_uses_the_raw_prompt_and_size():
    http, enhancer, renderer = client()
    response = http.post("/v1/images/generations",
                         json={"prompt": "exact", "enhance": False, "size": "1024x1024", "steps": 12})
    assert response.status_code == 200, response.text
    assert enhancer.calls == []
    assert renderer.calls[0] == {"prompt": "exact", "size": (1024, 1024), "seed": renderer.calls[0]["seed"],
                                 "steps": 12, "images": 0}


@pytest.mark.parametrize("payload, message", [
    ({}, "prompt is required"),
    ({"prompt": "x", "n": 2}, "n must be 1"),
    ({"prompt": "x", "response_format": "url"}, "b64_json"),
    ({"prompt": "x", "size": "1000x1000"}, "multiple of 32"),
    ({"prompt": "x", "steps": 0}, "steps"),
])
def test_generation_rejects_bad_requests(payload, message):
    http, _, renderer = client()
    response = http.post("/v1/images/generations", json=payload)
    assert response.status_code == 400
    assert message in response.json()["error"]["message"]
    assert renderer.calls == []


def test_edit_passes_images_and_follows_the_source_ratio():
    enhancer = FakeEnhancer({"positive_prompt": "edited", "wh_ratio": "", "ratio_follow": "<image1>", "parse_ok": True})
    http, enhancer, renderer = client(enhancer=enhancer)
    response = http.post("/v1/images/edits", data={"prompt": "make it night", "seed": "3"},
                         files=[("image[]", ("a.png", png_bytes((300, 200)), "image/png"))])
    assert response.status_code == 200, response.text
    assert enhancer.calls == [("edit", "make it night", 1, 3)]
    width, height = renderer.calls[0]["size"]
    assert renderer.calls[0]["images"] == 1 and width > height


def test_edit_rejects_masks_and_missing_images():
    http, _, _ = client()
    assert http.post("/v1/images/edits", data={"prompt": "x"}).status_code == 400
    masked = http.post("/v1/images/edits", data={"prompt": "x"},
                       files=[("image", ("a.png", png_bytes(), "image/png")),
                              ("mask", ("m.png", png_bytes(), "image/png"))])
    assert masked.status_code == 400 and "mask" in masked.json()["error"]["message"]


def test_renderer_failures_are_reported_not_fatal():
    class Broken(FakeRenderer):
        def render(self, *args, **kwargs):
            raise RuntimeError("CUDA out of memory")

    http, _, _ = client(renderer=Broken())
    response = http.post("/v1/images/generations", json={"prompt": "x", "enhance": False})
    assert response.status_code == 500 and "out of memory" in response.json()["error"]["message"]
    assert http.get("/health").status_code == 200


# --------------------------------------------------------------------------- #
# Renderer staging (no torch, no GPU: the pipeline and loaders are stand-ins)
# --------------------------------------------------------------------------- #
class StagePipe:
    """The slice of the pipeline the renderer's staging touches."""

    text_encoder = None
    transformer = None
    fail = False

    def register_modules(self, **modules):
        for name, module in modules.items():
            setattr(self, name, module)

    def encode_prompt(self, prompt):
        assert self.text_encoder is not None and self.transformer is None
        return "embeds"

    def __call__(self, **kwargs):
        assert self.encode_prompt(kwargs["prompt"]) == "embeds"
        assert self.text_encoder is None and self.transformer is not None
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return type("Output", (), {"images": ["image"]})()


def staged_renderer():
    class Generator:
        def __init__(self, device):
            pass

        def manual_seed(self, seed):
            return self

    renderer = object.__new__(server.QwenImageRenderer)
    renderer.pipe = StagePipe()
    renderer.torch = type("Torch", (), {
        "Generator": Generator, "cuda": type("Cuda", (), {"empty_cache": staticmethod(lambda: None)})})
    renderer.default_steps = None
    renderer.accepts = set()
    renderer.loads = []
    renderer._load = lambda name: renderer.loads.append(name) or object()
    renderer._wrap_encode_prompt()
    return renderer


def test_stage_keeps_one_large_model_loaded():
    renderer = staged_renderer()
    renderer._stage("text_encoder")
    assert renderer.pipe.text_encoder is not None and renderer.pipe.transformer is None
    renderer._stage("transformer")
    assert renderer.pipe.text_encoder is None and renderer.pipe.transformer is not None
    renderer._stage("transformer")  # already there: no reload
    assert renderer.loads == ["text_encoder", "transformer"]


def test_render_frees_both_models_and_reloads_next_time():
    renderer = staged_renderer()
    for _ in range(2):
        assert renderer.render("p", 1024, 1024, 7, None, []) == "image"
        assert renderer.pipe.text_encoder is None and renderer.pipe.transformer is None
    assert renderer.loads == ["text_encoder", "transformer"] * 2


def test_render_frees_both_models_when_the_pipeline_fails():
    renderer = staged_renderer()
    renderer.pipe.fail = True
    with pytest.raises(RuntimeError):
        renderer.render("p", 1024, 1024, 7, None, [])
    assert renderer.pipe.text_encoder is None and renderer.pipe.transformer is None
