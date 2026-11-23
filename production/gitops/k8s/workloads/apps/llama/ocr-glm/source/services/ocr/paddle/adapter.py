"""Pinned PP-DocLayoutV3 -> PaddleOCR-VL-1.6/vLLM, serial bounded regions.

The two internal predictors are deliberately bound to PaddleOCR 3.7.0 /
PaddleX 3.7.2. Calling them separately prevents the general pipeline's hidden
region fan-out and text/Markdown rewriting. No document-selected prompt exists.
"""

from __future__ import annotations

import importlib.metadata
import base64
import io
import math
import numbers
import os
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from .decode import MAX_SIDE, PixelBudget, ReferencePage
from .errors import WorkerError

EXPECTED_VERSIONS = {
    "paddleocr": "3.7.0", "paddlex": "3.7.2", "paddlepaddle": "3.2.1",
    "Pillow": "11.3.0", "pypdfium2": "4.30.0", "numpy": "2.2.6",
    "openai": "1.109.1",
}
MAX_REGIONS_PER_PAGE = 512
MAX_REGION_PIXELS = 1_605_632
MIN_REGION_PIXELS = 112_896
REGION_RESIZE_FACTOR = 28  # Pinned patch_size=14, merge_size=2.
MAX_NEW_TOKENS = 4096
MAX_SOURCE_REGION_PIXELS = 2 * MAX_REGION_PIXELS
MAX_INFERENCE_CALLS = 512
MAX_TOKEN_SPLIT_DEPTH = 3
MAX_RECOVERY_CALLS = 8
MAX_CONTEXT_REGIONS = 2
MAX_TABLE_TEXT_CONTEXTS = 2
MAX_IMAGE_TABLE_READS = 1
MAX_REFINEMENT_VIEWS = 4
MAX_REFINED_TABLE_READS = 2
OPTIONAL_CONTEXT_SECONDS = 240
MAX_TEXT = 1_000_000
MAX_BLOCK_TEXT = 10_000
MAX_BLOCKS = 50_000
PROMPT = "OCR:"
TABLE_PROMPT = "Table Recognition:"
# Exact per-class merge policy in PaddleX 3.7.2's PaddleOCR-VL-1.6.yaml.
# The old global "small" policy discarded parent invoice tables.
NATIVE_LAYOUT_MERGE = {index: "large" if index in (3, 5, 6, 15, 17) else "union"
    for index in range(25)}
KNOWN_LAYOUT_LABELS = frozenset((
    "abstract", "algorithm", "aside_text", "chart", "content", "display_formula",
    "doc_title", "figure_title", "footer", "footer_image", "footnote", "formula_number",
    "header", "header_image", "image", "inline_formula", "number", "paragraph_title",
    "reference", "reference_content", "seal", "table", "text", "vertical_text", "vision_footnote",
))


class OptionalContextDeadline(Exception):
    """Internal bounded-work signal, never serialized as provider output."""


@dataclass(frozen=True)
class WorkerConfig:
    layout_model_dir: str
    vl_model_dir: str
    vllm_base_url: str
    served_model_name: str
    pdf_scale: int = 2


def parse_config(value: object) -> WorkerConfig:
    keys = {"configVersion", "pipelineVersion", "layoutModelDir", "vlModelDir", "vllmBaseUrl", "servedModelName", "pdfScale", "layoutDevice"}
    if not isinstance(value, dict) or set(value) != keys:
        raise WorkerError("provider_unavailable")
    if type(value["configVersion"]) is not int or value["configVersion"] != 1 or value["pipelineVersion"] != "v1.6" or type(value["pdfScale"]) is not int or value["pdfScale"] != 2 or value["layoutDevice"] != "cpu" or value["servedModelName"] != "PaddleOCR-VL-1.6-0.9B":
        raise WorkerError("provider_unavailable")
    for key in ("layoutModelDir", "vlModelDir"):
        if not isinstance(value[key], str) or not os.path.isabs(value[key]) or not os.path.isdir(value[key]):
            raise WorkerError("provider_unavailable")
    try:
        url = urlsplit(value["vllmBaseUrl"])
        if url.scheme not in ("http", "https") or url.hostname not in ("ocr-vllm", "127.0.0.1", "::1") or not (1 <= (url.port or (443 if url.scheme == "https" else 80)) <= 65535) or url.username or url.password or url.query or url.fragment or url.path != "/v1":
            raise ValueError()
    except Exception:
        raise WorkerError("provider_unavailable") from None
    return WorkerConfig(value["layoutModelDir"], value["vlModelDir"], value["vllmBaseUrl"], value["servedModelName"])


def verify_versions() -> None:
    try:
        if any(importlib.metadata.version(name) != expected for name, expected in EXPECTED_VERSIONS.items()):
            raise WorkerError("provider_unavailable")
    except importlib.metadata.PackageNotFoundError:
        raise WorkerError("provider_unavailable") from None


def make_pipeline(config: WorkerConfig):
    verify_versions()
    # Require pre-staged model directories; offline flags prevent HF resolution.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    from paddleocr import PaddleOCRVL
    from paddlex.inference import load_pipeline_config
    import httpx

    pipeline_config = load_pipeline_config("PaddleOCR-VL-1.6")
    pipeline_config["batch_size"] = 1
    pipeline_config["SubModules"]["LayoutDetection"]["batch_size"] = 1
    recognition_config = pipeline_config["SubModules"]["VLRecognition"]
    recognition_config["batch_size"] = 1
    # These keys pass through PaddleX 3.7.2's GenAIClient to AsyncOpenAI.
    # No SDK retry, proxy from the environment, or redirect can submit a crop
    # somewhere outside the fixed private endpoint.
    recognition_config["genai_config"]["client_kwargs"] = {
        "max_retries": 0, "timeout": 60,
        "http_client": httpx.AsyncClient(follow_redirects=False, trust_env=False,
            timeout=60, limits=httpx.Limits(max_connections=1, max_keepalive_connections=1)),
    }

    return PaddleOCRVL(
        paddlex_config=pipeline_config,
        pipeline_version="v1.6", layout_detection_model_name="PP-DocLayoutV3",
        layout_detection_model_dir=config.layout_model_dir,
        vl_rec_model_name="PaddleOCR-VL-1.6-0.9B", vl_rec_model_dir=config.vl_model_dir,
        vl_rec_backend="vllm-server", vl_rec_server_url=config.vllm_base_url,
        vl_rec_api_model_name=config.served_model_name, vl_rec_max_concurrency=1,
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_layout_detection=True, use_chart_recognition=False,
        use_seal_recognition=False, use_ocr_for_image_block=True,
        format_block_content=False, merge_layout_blocks=False,
        markdown_ignore_labels=[], use_queues=False,
        device="cpu", enable_mkldnn=False, cpu_threads=2,
    )


def bounded_box(coordinates, width: int, height: int) -> tuple[int, int, int, int]:
    try:
        if len(coordinates) != 4 or any(isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v) for v in coordinates):
            raise ValueError()
        left, top, right, bottom = coordinates
        if left < 0 or top < 0 or right > width or bottom > height or right <= left or bottom <= top:
            raise ValueError()
        return math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)
    except Exception:
        raise WorkerError("invalid_provider_output") from None


def normalized_box(box: tuple[int, int, int, int], width: int, height: int):
    left, top, right, bottom = box
    x, y = left / width, top / height
    return {"x": x, "y": y, "width": min((right - left) / width, 1 - x), "height": min((bottom - top) / height, 1 - y)}


def resized_region_dimensions(height: int, width: int) -> tuple[int, int]:
    """Match the pinned model/PaddleX 3.7.2 smart_resize without allocating.

    The model's image_processing_paddleocr_vl.py and PaddleX's
    inference/genai/models/paddleocr_vl_09b/_vllm.py use this same algorithm.
    The latter only calculates token counts; it does not resize another image.
    Preserve Python's ties-to-even round and tiny-side enlargement BEFORE the
    aspect check. Merely clamping each side or using min_pixels is not a bound.
    """
    if any(type(side) is not int or side < 1 for side in (height, width)):
        raise WorkerError("invalid_provider_output")
    if max(height, width) > MAX_SIDE:
        raise WorkerError("resource_limit", resource_limit="region_dimensions")
    factor = REGION_RESIZE_FACTOR
    if height < factor:
        width = round(width * factor / height)
        height = factor
    if width < factor:
        height = round(height * factor / width)
        width = factor
    if max(height, width) / min(height, width) > 200:
        raise WorkerError("resource_limit", resource_limit="region_dimensions")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > MAX_REGION_PIXELS:
        scale = math.sqrt(height * width / MAX_REGION_PIXELS)
        resized_height = math.floor(height / scale / factor) * factor
        resized_width = math.floor(width / scale / factor) * factor
    elif resized_height * resized_width < MIN_REGION_PIXELS:
        scale = math.sqrt(MIN_REGION_PIXELS / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return resized_height, resized_width


def region_raster_work(width: int, height: int) -> int:
    """Conservative cumulative raster work for one serial RGB/PNG request.

    Source-sized upper charge (16): crop/RGB byte/Pillow copies (3), server
    decode (1), processor Pillow-to-NumPy bytes/array (3), resize's uint8/Pillow
    conversion (2), and vLLM's RGBA hash conversion/bytes/joins (up to 7).
    RGB conversions that are no-ops do not require a further copy.

    Resized upper charge (16): resize output plus Pillow bytes/NumPy array (4),
    float64 cast/multiply/float32 rescale (3), normalize subtract/divide (2),
    processor batch/tile/flatten/final arrays (4), and tensor contiguity/device/
    model-dtype copies (3). Channel transpose, unsqueeze and from_numpy are
    views. The bicubic separable intermediate is charged separately: rounding
    can enlarge one axis while shrinking the other, so max(source, target)
    area alone is not always a sufficient scratch bound.

    These charges follow the pinned model, Pillow 11.3, Transformers 4.55.4
    image_transforms.py, and vLLM 0.10.2 multimodal image/hasher/inputs.py paths.
    They count raster pixels, not channels/bytes or model feature activations;
    process/container memory limits independently bound those allocations.
    """
    resized_height, resized_width = resized_region_dimensions(height, width)
    scratch = max(width * resized_height, resized_width * height)
    return 16 * width * height + 16 * resized_width * resized_height + scratch


def layout_frame(width: int, height: int, target: tuple[int, int]) -> tuple[int, int]:
    scale = min(1, target[0] / width, target[1] / height)
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


def original_box(box, working_width: int, working_height: int, width: int, height: int):
    # Validate detector output in its actual input frame, then round outwards
    # into the unchanged, EXIF-oriented full-original reference frame.
    bounded_box(box, working_width, working_height)
    left, top, right, bottom = box
    return (math.floor(left * width / working_width), math.floor(top * height / working_height),
        min(width, math.ceil(right * width / working_width)), min(height, math.ceil(bottom * height / working_height)))


def recognition_frame(width: int, height: int) -> tuple[int, int]:
    resized_height, resized_width = resized_region_dimensions(height, width)
    # Preserve small original regions. The model still performs its normal
    # minimum-pixel resize; only oversized uploads are reduced ahead of PNG.
    return (resized_width, resized_height) if width * height > MAX_REGION_PIXELS else (width, height)


def recognition_raster_work(width: int, height: int) -> int:
    target_width, target_height = recognition_frame(width, height)
    scratch = max(width * target_height, target_width * height) if (target_width, target_height) != (width, height) else 0
    return region_raster_work(target_width, target_height) + scratch


def finer_image_views(box, width, height):
    """Fixed overlapping views; source text never chooses a crop or scale."""
    left, top, right, bottom = bounded_box(box, width, height)
    if right-left < 4 or bottom-top < 4:
        return []
    cx, cy = (left+right)//2, (top+bottom)//2
    ox, oy = max(1, (right-left)//8), max(1, (bottom-top)//8)
    boxes = [(left,top,cx+ox,cy+oy),(cx-ox,top,right,cy+oy),
             (left,cy-oy,cx+ox,bottom),(cx-ox,cy-oy,right,bottom)]
    views = []
    for source in boxes[:MAX_REFINEMENT_VIEWS]:
        w, h = source[2]-source[0], source[3]-source[1]
        # Check the same aspect/dimension domain before any allocation.
        resized_region_dimensions(h, w)
        scale = min(math.sqrt(MAX_REGION_PIXELS/(w*h)), MAX_SIDE/max(w,h))
        factor = REGION_RESIZE_FACTOR
        target = (max(factor,math.floor(w*scale/factor)*factor),
                  max(factor,math.floor(h*scale/factor)*factor))
        if (target[0]*target[1] > MAX_REGION_PIXELS or max(target) > MAX_SIDE
                or max(target)/min(target) > 200):
            raise WorkerError("resource_limit", resource_limit="region_dimensions")
        views.append((source,target))
    return views


def mapped_view_box(box, source, view_width, view_height):
    """Outward integer mapping into the unchanged full-original frame."""
    left, top, right, bottom = bounded_box(box, view_width, view_height)
    a,b,c,d = bounded_box(source, MAX_SIDE, MAX_SIDE)
    width, height = c-a, d-b
    return (a+left*width//view_width, b+top*height//view_height,
            min(c,a+(right*width+view_width-1)//view_width),
            min(d,b+(bottom*height+view_height-1)//view_height))


def whitespace_cut(page, box, budget):
    """Find a nearby quiet horizontal seam using bounded raster projection.

    Per-column light background discounts a constant dark table margin. This
    is an image heuristic, not text recognition or a semantic boundary proof.
    """
    import numpy as np
    from PIL import Image

    left, top, right, bottom = box
    span = bottom-top
    band_top, band_bottom = top+span*3//8, bottom-span*3//8
    width, height = right-left, band_bottom-band_top
    sample_width, sample_height = min(256,width), min(512,height)
    if sample_height < 4:
        return None
    # Source-ROI resize scratch plus RGB/L buffers, NumPy bytes/copies,
    # percentile scratch and projection/threshold arrays. Bound before resize.
    work = max(width*sample_height,sample_width*height)+12*sample_width*sample_height
    with budget.memory(work):
        budget.reserve(work)
        sample=page.image.resize((sample_width,sample_height),Image.Resampling.BOX,
            box=(left,band_top,right,band_bottom))
        try:
            grey=sample.convert("L")
            try:
                pixels=np.asarray(grey)
                background=np.percentile(pixels,85,axis=0)
                if float(np.median(background)) < 64:
                    return None
                scores=np.mean(pixels < (background[None,:]-20),axis=1)
                # A constant dark pattern (e.g. vertical rules/barcodes) is not
                # evidence of a blank seam merely because each column is
                # constant. Uniform near-white input is the safe blank case.
                if float(np.max(scores))==0 and int(np.min(pixels))<240:
                    return None
                runs=[];start=None
                for row in range(sample_height+1):
                    clean=row<sample_height and scores[row]==0
                    if clean and start is None:start=row
                    if not clean and start is not None:
                        if row-start>=2:runs.append((start,row))
                        start=None
                if not runs:return None
                middle=sample_height/2
                # A long blank run can contain the nominal midpoint itself.
                selected=min(runs,key=lambda run:max(run[0]-middle,middle-run[1],0))
                row=min(max(middle,selected[0]+.5),selected[1]-.5)
                return band_top+round(row*height/sample_height), max(2,math.ceil(height/sample_height)*2,span//20)
            finally:grey.close()
        finally:sample.close()


def split_region(box, page=None, budget=None, notes=None):
    """Overlapping original-coordinate crops; prefer a quiet printed-row seam."""
    left, top, right, bottom = box
    # Keep left labels and right-aligned amounts together whenever vertical
    # room permits; a near-one-line crop can only be split across its width.
    horizontal = bottom-top < 4
    start,end=(left,right) if horizontal else (top,bottom)
    span=end-start
    if span<4:return None
    seam=whitespace_cut(page,box,budget) if not horizontal and page is not None else None
    if seam is not None:
        middle,overlap=seam
    else:
        middle=start+span//2
        overlap=max(1,span//10)
    overlap=min(overlap,max(1,(span-1)//4),middle-start-1,end-middle-1)
    if notes is not None:notes.append("whitespace" if seam is not None else "fallback")
    if horizontal:
        return [(left,top,middle+overlap,bottom),(middle-overlap,top,right,bottom)]
    return [(left,top,right,middle+overlap),(left,middle-overlap,right,bottom)]


def tiled_regions(boxes, page=None, budget=None, notes=None):
    """Cover large layout regions without shrinking a whole phone photo.

    Quiet horizontal seams are preferred. Bounded overlap retains boundary
    context when no seam is found; this does not guarantee perfect OCR and
    repeated recognized text is retained rather than guessed away.
    """
    output=[]
    def add(box):
        left,top,right,bottom=box
        if (right-left)*(bottom-top)>MAX_SOURCE_REGION_PIXELS:
            for child in split_region(box,page,budget,notes):add(child)
        else:
            output.append(box)
            if len(output)>MAX_INFERENCE_CALLS:
                raise WorkerError("resource_limit",resource_limit="region_count")
    for box in boxes:add(box)
    return output


def context_regions(boxes, width, height, *, gap_percent=3):
    """At most two intact original ROIs joining nearby layout fragments.

    This is the prior adapter's rectangular context heuristic, not PaddleX's
    merged-image canvas or table pipeline. It never establishes document
    boundaries; complete-frame recognition remains independent of these ROIs.
    """
    if len(boxes)<2:return []
    parents=list(range(len(boxes)))
    def root(index):
        while parents[index]!=index:
            parents[index]=parents[parents[index]];index=parents[index]
        return index
    # Coordinates are already in the original frame.
    for i,(left,top,right,bottom) in enumerate(boxes):
        for j in range(i):
            a,b,c,d=boxes[j]
            overlap=min(right,c)-max(left,a)
            gap=max(0,top-d,b-bottom)
            if overlap*4>=min(right-left,c-a) and gap*100<=height*gap_percent:
                first,second=root(i),root(j)
                parents[max(first,second)]=min(first,second)
    groups={}
    for i in range(len(boxes)):groups.setdefault(root(i),[]).append(i)
    candidates=[]
    for indices in groups.values():
        if len(indices)<2:continue
        box=(min(boxes[i][0] for i in indices),min(boxes[i][1] for i in indices),
            max(boxes[i][2] for i in indices),max(boxes[i][3] for i in indices))
        if box not in candidates:candidates.append(box)
    candidates.sort(key=lambda b:(-(b[2]-b[0])*(b[3]-b[1]),b[1],b[0]))
    return sorted(candidates[:MAX_CONTEXT_REGIONS],key=lambda b:(b[1],b[0]))


def image_text_context_regions(regions, width, height):
    """Bounded substantial image views, selected without recognized text."""
    if len(regions) > MAX_REGIONS_PER_PAGE:
        raise WorkerError("resource_limit", resource_limit="region_count")
    # A screenshot's image can contain a document whose small text is lost in
    # the broad overview. Do not merge that detected image back into the outer
    # prose before recognition. Small icons and near-whole-frame detections
    # do not provide a useful independent text view. Selection uses geometry
    # and the native label only, never recognized words or financial values.
    images = []
    for label, (left, top, right, bottom) in sorted(regions,
            key=lambda item: (item[1][1], item[1][0], item[1][3], item[1][2])):
        area = (right-left)*(bottom-top)
        if (label != "image" or area*20 < width*height or area*5 >= width*height*4
                or (right-left)*4 < width or (bottom-top)*20 < height):
            continue
        # Outward padding preserves text close to a detected image boundary.
        # It remains an original-coordinate crop, subject to all read bounds.
        pad_x, pad_y = max(2, math.ceil((right-left)*.02)), max(2, math.ceil((bottom-top)*.02))
        box = (max(0, left-pad_x), max(0, top-pad_y),
               min(width, right+pad_x), min(height, bottom+pad_y))
        if box not in images and box != (0, 0, width, height):
            images.append(box)
    return images[:MAX_CONTEXT_REGIONS]


def ordinary_context_plan(regions, width, height):
    """Preserve separate text lanes and read substantial embedded images.

    This selects optional recognition context, never document boundaries.
    The unchanged whole-original overview remains the first mandatory read.
    Crops derive from actual original layout boxes, with bounded outward
    padding for embedded images.
    """
    if len(regions) > MAX_REGIONS_PER_PAGE:
        raise WorkerError("resource_limit", resource_limit="region_count")
    text_labels = frozenset((
        "abstract", "algorithm", "aside_text", "content", "doc_title",
        "figure_title", "footer", "footnote", "header", "number",
        "paragraph_title", "reference", "reference_content", "text", "vertical_text",
    ))
    boxes = [box for _, box in regions]
    text_boxes = [box for label, box in regions if label in text_labels]
    # Four percent joins a short header/body gap without admitting unrelated
    # image/chart regions. Use this only for a separated pair of substantial
    # side-by-side lanes; all other layouts keep the existing context plan.
    text_contexts = context_regions(text_boxes, width, height, gap_percent=4)
    if len(text_contexts) == 2:
        left, right = sorted(text_contexts, key=lambda box: box[0])
        widths = [box[2]-box[0] for box in (left, right)]
        heights = [box[3]-box[1] for box in (left, right)]
        horizontal_overlap = max(0, min(left[2], right[2])-max(left[0], right[0]))
        vertical_overlap = min(left[3], right[3])-max(left[1], right[1])
        centres_apart = (right[0]+right[2]-left[0]-left[2])/2
        if (all(width*.25 <= value <= width*.65 for value in widths)
                and all(height*.15 <= value <= height*.70 for value in heights)
                and centres_apart >= width*.25
                and horizontal_overlap <= min(widths)*.05
                and vertical_overlap >= min(heights)*.6):
            return text_contexts, True
    images = image_text_context_regions(regions, width, height)
    contexts = context_regions(boxes, width, height)
    return list(dict.fromkeys([*images, *contexts]))[:MAX_CONTEXT_REGIONS], False


def table_dominates(regions):
    """Tables cover a majority of the validated layout envelope, not the photo.

    Exact rectangle union prevents overlapping detections from inflating the
    table fraction. This is a scheduling heuristic, not document classification.
    """
    if len(regions) > MAX_REGIONS_PER_PAGE:
        raise WorkerError("resource_limit", resource_limit="region_count")
    tables = [box for label, box in regions if label == "table"]
    if not tables:
        return False
    boxes = [box for _, box in regions]
    area = ((max(box[2] for box in boxes) - min(box[0] for box in boxes))
        * (max(box[3] for box in boxes) - min(box[1] for box in boxes)))
    edges = sorted({coordinate for box in tables for coordinate in (box[0], box[2])})
    covered = 0
    for left, right in zip(edges, edges[1:]):
        intervals = sorted((top, bottom) for a, top, b, bottom in tables if a < right and b > left)
        height = 0
        end = None
        for top, bottom in intervals:
            if end is None or top > end:
                height += bottom - top
                end = bottom
            elif bottom > end:
                height += bottom - end
                end = bottom
        covered += (right - left) * height
    return covered * 2 >= area


def peak_fit_regions(box,page,budget,notes,depth=0):
    """Keep a whole context/overview unless its live raster bound cannot fit."""
    left,top,right,bottom=box
    work=recognition_raster_work(right-left,bottom-top)
    try:
        with budget.memory(work):pass
        return [box]
    except WorkerError as error:
        if error.resource_limit!="pixel_budget" or depth>=MAX_TOKEN_SPLIT_DEPTH:raise
        children=split_region(box,page,budget,notes)
        if children is None:raise
        return [part for child in children for part in peak_fit_regions(child,page,budget,notes,depth+1)]


def native_center_region(box, width, height):
    """Full-width source center sample under the existing model pixel cap.

    It supplements retained stopped native table evidence; this partial band
    does not claim whole-ROI coverage. Geometry never inspects text or values.
    Upload dimensions stay unchanged; internal model patch alignment may resize.
    """
    left, top, right, bottom = bounded_box(box, width, height)
    band_height = min(bottom-top, MAX_REGION_PIXELS//(right-left))
    if band_height < 1:
        raise WorkerError("resource_limit", resource_limit="region_dimensions")
    band_top = top+(bottom-top-band_height)//2
    return left, band_top, right, band_top+band_height


def _one_result(results):
    iterator = iter(results)
    try:
        first = next(iterator)
        sentinel = object()
        if next(iterator, sentinel) is not sentinel:
            raise WorkerError("invalid_provider_output")
        return first
    except StopIteration:
        raise WorkerError("invalid_provider_output") from None
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()


def unreadable(page: ReferencePage, message: str):
    warning = {"code": "page_unreadable", "message": message, "page": page.number}
    return {"page": page.number, "width": page.width, "height": page.height,
        "status": "unreadable", "reason": message, "text": "", "blocks": [], "warnings": [warning]}


class PaddleAdapter:
    def __init__(self, config: WorkerConfig, budget: PixelBudget, pipeline_factory=make_pipeline):
        self.budget = budget
        self.text_used = 0
        self.block_count = 0
        self.inference_calls = 0
        self.recovery_calls = 0
        self.optional_deadline = time.monotonic()+OPTIONAL_CONTEXT_SECONDS
        self.token_split_used = False
        self.split_notes = []
        try:
            self.pipeline = pipeline_factory(config)
            wrapper = self.pipeline.paddlex_pipeline
            if wrapper.multi_device_inference:
                raise WorkerError("provider_unavailable")
            internal = wrapper._pipeline
            self.layout = internal.layout_det_model
            self.recognizer = internal.vl_rec_model
            if not callable(self.layout) or not callable(self.recognizer.genai_client.create_chat_completion):
                raise WorkerError("provider_unavailable")
            # The pinned native layout predictor exposes its actual resize
            # target. Unknown/oversized preprocessing must not escape the budget.
            target = getattr(self.layout, "target_size", None)
            if not isinstance(target, (list, tuple)) or len(target) != 2 or any(type(v) is not int or v < 1 or v > 1024 for v in target):
                raise WorkerError("provider_unavailable")
            self.layout_pixels = target[0] * target[1]
            self.layout_target = tuple(target)
        except WorkerError:
            raise
        except Exception:
            raise WorkerError("provider_unavailable") from None

    def close(self):
        self.pipeline.close()

    def recognize_crop(self, crop, task=PROMPT):
        if task not in (PROMPT, TABLE_PROMPT):
            raise WorkerError("invalid_provider_output")
        # PaddleX's generic predictor JPEG-encodes crops and drops finish_reason.
        # Its pinned GenAIClient preserves the complete response, while lossless
        # PNG avoids adding compression artifacts to original raster evidence.
        with io.BytesIO() as encoded:
            crop.save(encoded, format="PNG")
            data_uri = "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")
        response = self.recognizer.genai_client.create_chat_completion(
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": task},
            ]}], return_future=False, timeout=60,
            max_completion_tokens=MAX_NEW_TOKENS, temperature=0, top_p=1,
            extra_body={"skip_special_tokens": True, "repetition_penalty": 1,
                "mm_processor_kwargs": {"min_pixels": MIN_REGION_PIXELS, "max_pixels": MAX_REGION_PIXELS}},
        )
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or len(choices) != 1:
            raise WorkerError("invalid_provider_output")
        choice = choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise WorkerError("resource_limit", resource_limit="token_limit")
        message = getattr(choice, "message", None)
        if (getattr(choice, "finish_reason", None) != "stop" or message is None
                or getattr(message, "tool_calls", None) or getattr(message, "function_call", None)
                or getattr(message, "refusal", None)):
            raise WorkerError("invalid_provider_output")
        return getattr(message, "content", None)

    def read_region(self, page, box, depth=0, optional_deadline=None, recover=True, task=PROMPT):
        """Serial bounded read; discard length-finished text before recovery."""
        from PIL import Image

        # Do not start a sixty-second optional request when it cannot finish
        # within the reserved window. Leave sixty seconds before the unchanged
        # worker alarm for cleanup/protocol delivery and any earlier decoding.
        if optional_deadline is not None and time.monotonic()+60>optional_deadline:
            raise OptionalContextDeadline()
        if self.inference_calls >= MAX_INFERENCE_CALLS:
            raise WorkerError("resource_limit", resource_limit="region_count")
        if depth and self.recovery_calls>=MAX_RECOVERY_CALLS:
            raise WorkerError("resource_limit",resource_limit="token_limit")
        left, top, right, bottom = box
        target = recognition_frame(right-left, bottom-top)
        work = recognition_raster_work(right-left, bottom-top)
        try:
            with self.budget.memory(work):
                self.budget.reserve(work)
                crop = page.image.resize(target, Image.Resampling.BICUBIC, box=(left, top, right, bottom)) if target != (right-left, bottom-top) else page.image.crop((left, top, right, bottom))
                try:
                    self.inference_calls += 1
                    if depth:self.recovery_calls += 1
                    text = self.recognize_crop(crop, task)
                finally:
                    crop.close()
            if not isinstance(text, str):
                raise WorkerError("invalid_provider_output")
            try:
                text.encode("utf-8", "strict")
            except UnicodeError:
                raise WorkerError("invalid_provider_output") from None
            if len(text) > MAX_BLOCK_TEXT:
                raise WorkerError("resource_limit", resource_limit="block_text")
            return [(box, text)]
        except WorkerError as error:
            if not recover or error.resource_limit != "token_limit" or depth >= MAX_TOKEN_SPLIT_DEPTH:
                raise
            children = split_region(box,page,self.budget,self.split_notes)
            if children is None:
                raise
            self.token_split_used = True
            # No truncated parent text is published. If any child still fails,
            # the caller receives a failure, never an apparently complete page.
            return [item for child in children for item in self.read_region(page, child, depth + 1, optional_deadline, recover, task)]

    def retain_reads(self,page,reads,blocks,task=PROMPT):
        """Commit only whole validated reads within the output allowance."""
        if task not in (PROMPT, TABLE_PROMPT):
            raise WorkerError("invalid_provider_output")
        nonempty=[(box,text) for box,text in reads if text.strip()]
        needed=sum(len(text)+1 for _,text in nonempty)-(1 if nonempty and not blocks else 0)
        if self.block_count+len(nonempty)>MAX_BLOCKS or self.text_used+needed>MAX_TEXT:
            raise WorkerError("resource_limit",resource_limit="total_text")
        for box,text in nonempty:
            prefix = "table-" if task == TABLE_PROMPT else ""
            blocks.append({"id":f"{prefix}b{len(blocks)+1}","text":text,
                "box":normalized_box(box,page.width,page.height)})
        self.block_count+=len(nonempty)
        self.text_used+=needed
        return len(reads)-len(nonempty)

    @staticmethod
    def order_blocks(blocks):
        blocks.sort(key=lambda block: (block["box"]["y"], block["box"]["x"]))
        for index, block in enumerate(blocks, 1):
            # Only retain_reads assigns this internal task-derived prefix.
            prefix = "table-" if block["id"].startswith("table-") else ""
            block["id"] = f"{prefix}b{index}"

    def recognize_native_tables(self, page, regions, *, mixed_layout=False):
        """Native task dispatch with bounded, independently retained recovery.

        Table parents are attempted before other native elements. A length
        result is discarded, then exactly two original-coordinate children
        may each run once; no child recursively splits or repairs markup.
        """
        blocks = []
        incomplete = 0
        recovered = False
        last_error = None
        context_attempted = False
        pair_incomplete = False
        text_context_incomplete = False
        table_sources = []
        text_context_planned = 0
        text_context_cropped = False
        ordered = sorted(regions, key=lambda item: (item[0] != "table", item[1][1], item[1][0]))

        def attempt(box, task, *, child=False):
            reads = self.read_region(page, box, depth=1 if child else 0,
                optional_deadline=self.optional_deadline, recover=False, task=task)
            empty = self.retain_reads(page, reads, blocks, task)
            if task == TABLE_PROMPT:
                for source_box, text in reads:
                    if text.strip() and source_box not in table_sources:
                        table_sources.append(source_box)
            return empty == 0

        def ordinary_pair():
            nonlocal context_attempted, pair_incomplete, last_error
            context_attempted = True
            full = (0, 0, page.width, page.height)
            contexts = [box for box in context_regions([box for _, box in regions], page.width, page.height)
                if box != full]
            # The whole original and intact connected contexts retain the
            # label/value and issuer relationships lost by native fragments.
            # Each gets one attempt; no generic halves or recursive fan-out.
            pair_incomplete = not contexts
            for box in [full, *contexts]:
                try:
                    if not attempt(box, PROMPT):
                        pair_incomplete = True
                except (OptionalContextDeadline, WorkerError) as error:
                    if isinstance(error, WorkerError) and error.code != "resource_limit":
                        raise
                    pair_incomplete = True
                    if isinstance(error, WorkerError):
                        last_error = error

        def table_text_context():
            nonlocal context_attempted, text_context_incomplete, text_context_planned, text_context_cropped, last_error
            context_attempted = True
            # Only actual retained stopped table ROIs are eligible. A length
            # parent is absent; its completed children retain their own boxes.
            selected = sorted(table_sources, key=lambda box: (box[1], box[0]))[:MAX_TABLE_TEXT_CONTEXTS]
            text_context_planned = len(selected)
            for box in selected:
                try:
                    center = native_center_region(box, page.width, page.height)
                    cropped = center != box
                    text_context_cropped = text_context_cropped or cropped
                    if cropped and MAX_TOKEN_SPLIT_DEPTH < 1:
                        raise WorkerError("resource_limit", resource_limit="token_limit")
                    # Partial source samples conservatively use the unchanged
                    # global recovery allowance; no context retry or fan-out.
                    if not attempt(center, PROMPT, child=cropped):
                        text_context_incomplete = True
                except (OptionalContextDeadline, WorkerError) as error:
                    if isinstance(error, WorkerError) and error.code != "resource_limit":
                        raise
                    text_context_incomplete = True
                    if isinstance(error, WorkerError):
                        last_error = error

        def supplement_context():
            if mixed_layout:
                ordinary_pair()
            else:
                table_text_context()

        for label, box in ordered:
            if label != "table" and not context_attempted:
                supplement_context()
            task = TABLE_PROMPT if label == "table" else PROMPT
            try:
                complete = attempt(box, task)
            except (OptionalContextDeadline, WorkerError) as error:
                if isinstance(error, WorkerError) and error.code != "resource_limit":
                    raise
                complete = False
                if isinstance(error, WorkerError):
                    last_error = error
                if task == TABLE_PROMPT and isinstance(error, WorkerError) and error.resource_limit == "token_limit":
                    try:
                        # Before scanning, reserve enough time for at least one
                        # bounded request. Each child rechecks every limit.
                        if box[3] - box[1] < 4 or MAX_TOKEN_SPLIT_DEPTH < 1:
                            raise OptionalContextDeadline()
                        if time.monotonic() + 60 > self.optional_deadline:
                            raise OptionalContextDeadline()
                        children = split_region(box, page, self.budget, self.split_notes)
                    except (OptionalContextDeadline, WorkerError) as split_error:
                        if isinstance(split_error, WorkerError) and split_error.code != "resource_limit":
                            raise
                        children = None
                    successes = 0
                    for child_box in children or []:
                        try:
                            if attempt(child_box, task, child=True):
                                successes += 1
                        except (OptionalContextDeadline, WorkerError) as child_error:
                            if isinstance(child_error, WorkerError) and child_error.code != "resource_limit":
                                raise
                            if isinstance(child_error, WorkerError):
                                last_error = child_error
                    complete = successes == 2
                    recovered = recovered or successes > 0
            if not complete:
                incomplete += 1

        if not context_attempted:
            supplement_context()
        self.order_blocks(blocks)
        message = ("Fixed native table/OCR tasks used original-coordinate regions. "
            "Layout may miss text; fine detail and multiple documents need review. No separation was inferred.")
        if mixed_layout:
            message += " One bounded full-original overview and up to two intact connected original contexts were also attempted."
        elif text_context_planned:
            message += " Independent OCR context was planned for up to two completed table ROIs."
        if text_context_cropped:
            message += " Large-ROI OCR center samples do not cover entire ROIs."
        if recovered:
            message += " Truncated table parents were discarded; completed overlapping child reads were retained."
        if "fallback" in self.split_notes:
            message += " Some table boundaries had no reliable whitespace seam; overlap can still lose text context."
        warnings = [{"code": "low_quality", "page": page.number, "message": message}]
        if incomplete or pair_incomplete or text_context_incomplete:
            reason = f"{incomplete} of {len(regions)} validated native regions did not complete or returned no text." if incomplete else ""
            if pair_incomplete:
                reason += " The mixed-layout overview/context pair did not fully complete."
            if text_context_incomplete:
                reason += " Independent OCR context for completed table ROIs did not fully complete."
            warnings.append({"code": "page_unreadable", "page": page.number,
                "message": reason.strip() + " Only completed source-backed reads within the time, work, call and text limits were retained; original coverage is incomplete."})
        if not blocks:
            if last_error is not None:
                raise last_error
            return unreadable(page, "No supported text could be recognized in the native layout regions.")
        return {"page": page.number, "width": page.width, "height": page.height,
            "status": "read", "reason": None, "text": "\n".join(block["text"] for block in blocks),
            "blocks": blocks, "warnings": warnings}

    def detect_layout_regions(self, page):
        """Reuse the pinned bounded layout pass in this reference frame."""
        import numpy as np
        from PIL import Image
        working_width, working_height = layout_frame(page.width, page.height, self.layout_target)
        reduced = (working_width, working_height) != (page.width, page.height)
        resize_work = working_width * working_height + max(page.width * working_height, working_width * page.height) if reduced else 0
        layout_work = resize_work + 3 * working_width * working_height + 6 * self.layout_pixels
        # Layout buffers are released before even the first region read.
        with self.budget.memory(layout_work):
            self.budget.reserve(layout_work)
            working = page.image.resize((working_width, working_height), Image.Resampling.BICUBIC) if reduced else page.image
            pixels = None
            try:
                pixels = np.asarray(working)[:, :, ::-1].copy()
                layout_result = _one_result(self.layout([pixels], threshold=0.3,
                    layout_nms=True, layout_unclip_ratio=[1.0, 1.0],
                    layout_merge_bboxes_mode=dict(NATIVE_LAYOUT_MERGE), layout_shape_mode="rect",
                    filter_overlap_boxes=True))
                boxes = layout_result.get("boxes")
                if not isinstance(boxes, list):
                    raise WorkerError("invalid_provider_output")
                if len(boxes) > MAX_REGIONS_PER_PAGE:
                    raise WorkerError("resource_limit", resource_limit="region_count")
                original_regions=[]
                for box in boxes:
                    if (not isinstance(box, dict) or "coordinate" not in box
                            or not isinstance(box.get("label"), str) or box["label"] not in KNOWN_LAYOUT_LABELS):
                        raise WorkerError("invalid_provider_output")
                    mapped = original_box(box["coordinate"], working_width, working_height, page.width, page.height)
                    original_regions.append((box["label"], mapped))
                del layout_result
            finally:
                if pixels is not None:
                    pixels.fill(0)
                    del pixels
                if reduced:
                    working.close()
        return original_regions

    def refine_image_tables(self, page, box, blocks, region_count):
        """Inspect fixed magnified source views, retaining at most two tables.

        Layout alone chooses the nested ROIs. Original reads stay intact;
        completed table evidence is mapped outwards into the original frame.
        Views are serial, nonrecursive and share every document-wide bound.
        """
        from PIL import Image

        stats = {"views": 0, "attempts": 0, "reads": 0,
                 "incomplete": False, "limited": False}
        try:
            views = finer_image_views(box, page.width, page.height)
        except WorkerError as error:
            if error.code != "resource_limit":
                raise
            stats["limited"] = True
            return stats
        for source, target in views:
            if stats["attempts"] >= MAX_REFINED_TABLE_READS:
                stats["limited"] = True
                break
            if (time.monotonic()+60 > self.optional_deadline
                    or self.inference_calls >= MAX_INFERENCE_CALLS
                    or self.recovery_calls >= MAX_RECOVERY_CALLS
                    or MAX_TOKEN_SPLIT_DEPTH < 1):
                stats["limited"] = True
                break
            width, height = source[2]-source[0], source[3]-source[1]
            view_work = target[0]*target[1]+max(width*target[1], target[0]*height)
            try:
                with self.budget.memory(view_work):
                    self.budget.reserve(view_work)
                    view = page.image.resize(target, Image.Resampling.BICUBIC, box=source)
                    virtual = ReferencePage(page.number, *target, view)
                    try:
                        view.info.clear()
                        regions = self.detect_layout_regions(virtual)
                        stats["views"] += 1
                        region_count += len(regions)
                        if region_count > MAX_REGIONS_PER_PAGE:
                            stats["limited"] = True
                            break
                        tables = sorted((b for label, b in regions if label == "table"),
                            key=lambda b: (b[1], b[0]))
                        for table in tables:
                            if stats["attempts"] >= MAX_REFINED_TABLE_READS:
                                stats["incomplete"] = stats["limited"] = True
                                break
                            stats["attempts"] += 1
                            try:
                                reads = self.read_region(virtual, table, depth=1,
                                    optional_deadline=self.optional_deadline,
                                    recover=False, task=TABLE_PROMPT)
                                mapped = [(mapped_view_box(b, source, *target), text)
                                    for b, text in reads]
                                empty = self.retain_reads(page, mapped, blocks, TABLE_PROMPT)
                                stats["reads"] += len(reads)-empty
                                stats["incomplete"] |= not reads or empty > 0
                            except (OptionalContextDeadline, WorkerError) as error:
                                if isinstance(error, WorkerError) and error.code != "resource_limit":
                                    raise
                                stats["incomplete"] = stats["limited"] = True
                    finally:
                        virtual.close()
            except (OptionalContextDeadline, WorkerError) as error:
                if isinstance(error, WorkerError) and error.code != "resource_limit":
                    raise
                stats["limited"] = True
                break
        return stats

    def recognize(self, page: ReferencePage) -> dict:
        if page.image is None:
            return unreadable(page, page.reason or "The page could not be rendered.")
        from PIL import Image

        self.token_split_used = False
        self.split_notes = []
        try:
            original_regions = self.detect_layout_regions(page)
            if any(label == "table" for label, _ in original_regions):
                return self.recognize_native_tables(page, original_regions,
                    mixed_layout=not table_dominates(original_regions))
            # One bounded full-frame overview avoids excluding undetected
            # content without spending every job on mandatory full-res bands.
            # Supplementary intact context ROIs preserve receipt label/value
            # relationships that fragmented crops or bands can lose.
            page_warnings=[{"code":"low_quality","page":page.number,
                "message":"A full reference overview was read at bounded model resolution; fine text may be missed. Multiple documents may require human review; document separation was not inferred."}]
            full_box=(0,0,page.width,page.height)
            overview_error=None
            try:
                full_work=recognition_raster_work(page.width,page.height)
                with self.budget.memory(full_work):pass
                self.budget.check_work(full_work)
                if self.inference_calls>=MAX_INFERENCE_CALLS:
                    raise WorkerError("resource_limit",resource_limit="region_count")
                planned=[full_box]
            except WorkerError as error:
                if error.code!="resource_limit":raise
                overview_error=error
                planned=[]
            coverage_count=len(planned)
            costs=[recognition_raster_work(r-l,b-t) for l,t,r,b in planned]
            self.budget.check_work(sum(costs))
            if self.inference_calls+len(planned)>MAX_INFERENCE_CALLS:
                raise WorkerError("resource_limit",resource_limit="region_count")
            skipped_context=False
            contexts, paired_contexts = ordinary_context_plan(original_regions,page.width,page.height)
            for candidate in contexts:
                if candidate==full_box or candidate in planned:continue
                try:
                    left,top,right,bottom=candidate
                    extra=recognition_raster_work(right-left,bottom-top)
                    # Optional planning must not spend scan/resize work already
                    # needed by mandatory coverage. A context that cannot fit
                    # intact is skipped, rather than starting speculative bands.
                    with self.budget.memory(extra):pass
                    self.budget.check_work(sum(costs)+extra)
                    if self.inference_calls+len(planned)+1>MAX_INFERENCE_CALLS:
                        raise WorkerError("resource_limit",resource_limit="region_count")
                except WorkerError as error:
                    if error.code!="resource_limit":raise
                    skipped_context=True;continue
                planned.append(candidate);costs.append(extra)
            # The overview always executes before optional contexts, preserving
            # coverage headroom. Emitted evidence is subsequently ordered by
            # original coordinates, never by a preferred financial outcome.
            blocks = []
            empty_regions = 0
            read_contexts=0
            paired_context_successes=0
            repeated_frame_skipped=False
            fallback_coverage=False
            fallback_attempted=False
            def attempt_broad_fallback():
                nonlocal fallback_coverage,fallback_attempted,empty_regions,skipped_context
                fallback_attempted=True
                if page.height<4:return
                # After any incomplete overview, prioritize exactly
                # two full-width overlapping halves. Neither half is recursively
                # retried: a length-finished overview must not consume the job's
                # entire time allowance before broad source evidence is retained.
                if time.monotonic()+60<=self.optional_deadline:
                    try:
                        halves=split_region(full_box,page,self.budget,self.split_notes)
                    except WorkerError as error:
                        if error.code!="resource_limit":raise
                        halves=None
                    successes=0
                    for half in halves or []:
                        split_before=self.token_split_used;notes_before=len(self.split_notes)
                        try:
                            reads=self.read_region(page,half,optional_deadline=self.optional_deadline,recover=False)
                            empty_regions+=self.retain_reads(page,reads,blocks)
                            if any(text.strip() for _,text in reads):successes+=1
                        except (OptionalContextDeadline,WorkerError) as error:
                            if isinstance(error,WorkerError) and error.code!="resource_limit":raise
                            skipped_context=True
                            self.token_split_used=split_before
                            del self.split_notes[notes_before:]
                    fallback_coverage=successes==2
            for index,box in enumerate(planned):
                is_overview=index<coverage_count
                if not is_overview and overview_error is not None and not fallback_attempted:
                    attempt_broad_fallback()
                if not is_overview and overview_error is not None:
                    left,top,right,bottom=box
                    # Repeating almost the same failed frame is not a useful
                    # context fallback. Preserve time for genuinely narrower
                    # source-backed regions rather than another length-limited request.
                    if (right-left)*(bottom-top)*10>=page.width*page.height*9:
                        repeated_frame_skipped=True
                        continue
                split_before=self.token_split_used;notes_before=len(self.split_notes)
                try:
                    reads=self.read_region(page,box,optional_deadline=self.optional_deadline if not is_overview else None,
                        recover=not is_overview and overview_error is None)
                    empty_regions+=self.retain_reads(page,reads,blocks)
                    if paired_contexts and not is_overview and reads and all(text.strip() for _,text in reads):
                        paired_context_successes+=1
                except (OptionalContextDeadline,WorkerError) as error:
                    if isinstance(error,WorkerError) and error.code!="resource_limit":
                        raise
                    if is_overview:
                        if not isinstance(error,WorkerError):raise
                        overview_error=error
                    else:
                        skipped_context=True
                    self.token_split_used=split_before
                    del self.split_notes[notes_before:]
                    continue
                if index>=coverage_count:read_contexts+=1
            if overview_error is not None and not fallback_attempted:
                attempt_broad_fallback()
            # Keep every completed ordinary reading. A detector-located image
            # can contain a table even when its native label is "image";
            # one independent fixed table task can recover columns omitted by
            # the ordinary task. No recognized text chooses the task or ROI.
            image_table_incomplete = False
            image_tables = [] if paired_contexts else image_text_context_regions(
                original_regions, page.width, page.height)[:MAX_IMAGE_TABLE_READS]
            for box in image_tables:
                try:
                    reads = self.read_region(page, box, optional_deadline=self.optional_deadline,
                        recover=False, task=TABLE_PROMPT)
                    empty = self.retain_reads(page, reads, blocks, TABLE_PROMPT)
                    empty_regions += empty
                    image_table_incomplete = image_table_incomplete or not reads or empty > 0
                except (OptionalContextDeadline, WorkerError) as error:
                    if isinstance(error, WorkerError) and error.code != "resource_limit":
                        raise
                    image_table_incomplete = True
            refinement = None
            if image_tables:
                refinement = self.refine_image_tables(page, image_tables[0], blocks, len(original_regions))
            self.order_blocks(blocks)
            if overview_error is not None:
                if not blocks:raise overview_error
                if fallback_coverage:
                    page_warnings[0]={"code":"low_quality","page":page.number,
                        "message":"The full-reference overview did not complete; two bounded overlapping full-width reads covered its original frame. No unfinished overview text was used."}
                else:
                    ending="Truncated overview text was discarded." if overview_error.resource_limit=="token_limit" else "No unfinished overview text was used."
                    page_warnings[0]={"code":"page_unreadable","page":page.number,
                        "message":"The full-reference overview did not complete; only bounded context regions were read. Whole-original coverage is incomplete. "+ending}
            if repeated_frame_skipped:
                page_warnings.append({"code":"low_quality","page":page.number,
                    "message":"A near full-frame context was not repeated after the overview failed; completed source-backed fallback or context reads were retained."})
            if read_contexts:
                page_warnings.append({"code":"low_quality","page":page.number,
                    "message":"Additional original regions were read for context; overlapping recognition can disagree and requires review."})
            if image_tables:
                incomplete = image_table_incomplete or refinement["incomplete"]
                message = ("Embedded-image table supplements retained the earlier ordinary reads. "
                    f"Fixed magnified views checked: {refinement['views']}; nested table reads retained: {refinement['reads']}. "
                    "Raw readings may disagree, layout can miss fine text, and full table coverage is not established.")
                if incomplete:
                    message += " A detected supplementary table read was empty or did not complete."
                elif refinement["limited"]:
                    message += " Further optional work was bounded."
                page_warnings.append({"code":"page_unreadable" if incomplete else "low_quality",
                    "page":page.number,"message":message})
            if skipped_context:
                page_warnings.append({"code":"low_quality","page":page.number,
                    "message":"Some supplementary context crops could not fit the remaining bounded time, work, text or call allowance; successful source-backed reads were retained."})
            if paired_contexts and paired_context_successes < 2:
                page_warnings.append({"code":"page_unreadable","page":page.number,
                    "message":"Both independently planned side-by-side source contexts did not complete with readable text; boundary interpretation remains incomplete. The original overview and successful reads are retained."})
            if self.token_split_used:
                page_warnings.append({"code": "low_quality", "page": page.number,
                    "message": "A truncated recognition was discarded and its original region was read again as bounded overlapping crops."})
            if "fallback" in self.split_notes:
                page_warnings.append({"code":"low_quality","page":page.number,
                    "message":"Some crop boundaries had no reliable whitespace seam; overlapping reads preserve pixels but can still lose text context."})
            if not blocks:
                extrema = page.image.getextrema()
                white_blank = all(low == high and low >= 250 for low, high in extrema)
                if not white_blank:
                    return unreadable(page, "No supported text could be recognized on this page.")
            elif empty_regions:
                page_warnings.append({"code": "low_quality", "page": page.number,
                    "message": "Some overview or context reads produced no recognized text."})
            return {"page": page.number, "width": page.width, "height": page.height,
                "status": "read", "reason": None, "text": "\n".join(item["text"] for item in blocks),
                "blocks": blocks, "warnings": page_warnings}
        except WorkerError:
            raise
        except (ConnectionError, TimeoutError):
            raise WorkerError("provider_unavailable", True) from None
        except MemoryError:
            raise WorkerError("resource_limit", resource_limit="memory") from None
        except Exception:
            raise WorkerError("provider_failure", True) from None
