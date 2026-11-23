"""Decode only bounded original bytes. Never inspect PDF text or follow links."""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import stat
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from .errors import WorkerError

MAX_INPUT_BYTES = 10_485_760
MAX_PAGES = 50
MAX_SIDE = 16_000
MAX_PAGE_PIXELS = 40_000_000
MAX_TOTAL_PIXELS = 100_000_000
# Separate finite bounds: decoded source area, simultaneously live raster
# equivalents, and cumulative work are different resources. Serially releasing
# a crop must not make every later crop consume the same memory allowance.
MAX_RASTER_WORK = 2_000_000_000
MIMES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def dimensions(width, height) -> tuple[int, int]:
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, float))
        or not isinstance(height, (int, float))
        or not math.isfinite(width)
        or not math.isfinite(height)
        or width <= 0
        or height <= 0
    ):
        raise WorkerError("invalid_document")
    width, height = math.ceil(width), math.ceil(height)
    if width > MAX_SIDE or height > MAX_SIDE or width * height > MAX_PAGE_PIXELS:
        raise WorkerError("resource_limit", resource_limit="page_dimensions")
    return width, height


@dataclass
class PixelBudget:
    used: int = 0
    live: int = 0
    peak: int = 0

    def check_work(self, pixels: int) -> None:
        if type(pixels) is not int or pixels < 0 or self.used + pixels > MAX_RASTER_WORK:
            raise WorkerError("resource_limit", resource_limit="pixel_budget")

    def reserve(self, pixels: int) -> None:
        self.check_work(pixels)
        self.used += pixels

    def hold(self, pixels: int) -> None:
        if type(pixels) is not int or pixels < 0 or self.live + pixels > MAX_TOTAL_PIXELS:
            raise WorkerError("resource_limit", resource_limit="pixel_budget")
        self.live += pixels
        self.peak = max(self.peak, self.live)

    def release(self, pixels: int) -> None:
        if type(pixels) is not int or pixels < 0 or pixels > self.live:
            raise WorkerError("provider_failure")
        self.live -= pixels

    @contextmanager
    def memory(self, pixels: int):
        """Preflight a conservative live bound before allocation; always release."""
        self.hold(pixels)
        try:
            yield
        finally:
            self.release(pixels)

    def raster(self, width, height) -> tuple[int, int]:
        width, height = dimensions(width, height)
        self.reserve(width * height)
        return width, height


@dataclass
class ReferencePage:
    number: int
    width: int
    height: int
    # PIL RGB image, owned by this page; only one raster is yielded at a time.
    image: object | None
    reason: str | None = None
    budget: PixelBudget | None = None

    def close(self):
        if self.image is not None:
            self.image.close()
            self.image = None
            if self.budget is not None:
                self.budget.release(self.width * self.height)
                self.budget = None


def read_original(filename: str, source: dict) -> bytes:
    """filename is exclusively a trusted parent-process argument, never document JSON."""
    if not isinstance(source, dict) or set(source) != {"mime", "sizeBytes", "originalSha256"}:
        raise WorkerError("invalid_document")
    size, sha, mime = source["sizeBytes"], source["originalSha256"], source["mime"]
    if (
        type(size) is not int or size < 1 or size > MAX_INPUT_BYTES
        or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
        or not isinstance(mime, str) or mime not in {*MIMES.values(), "application/pdf"}
        or not os.path.isabs(filename)
    ):
        raise WorkerError("invalid_document")
    try:
        descriptor = os.open(filename, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as original:
            info = os.fstat(original.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise WorkerError("invalid_document")
            if info.st_size != size:
                raise WorkerError("invalid_document")
            data = original.read(MAX_INPUT_BYTES + 1)
            after = os.fstat(original.fileno())
            if (after.st_size, after.st_mtime_ns, after.st_ino) != (info.st_size, info.st_mtime_ns, info.st_ino):
                raise WorkerError("invalid_document")
    except WorkerError:
        raise
    except OSError:
        raise WorkerError("invalid_document") from None
    if len(data) != size:
        raise WorkerError("invalid_document")
    if hashlib.sha256(data).hexdigest() != sha:
        raise WorkerError("invalid_document")
    if mime == "application/pdf" and not data.startswith(b"%PDF-"):
        raise WorkerError("invalid_document")
    return data


def _rgb(image, budget: PixelBudget):
    """Flatten transparency on white without changing the original reference frame."""
    from PIL import Image

    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        # RGBA conversion, white background, composite output, RGB conversion.
        # Reserve all four before starting even the first full-size allocation.
        work = 4 * image.width * image.height
        with budget.memory(work):
            budget.reserve(work)
            rgba = image.convert("RGBA")
            try:
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                try:
                    background.alpha_composite(rgba)
                    result = background.convert("RGB")
                finally:
                    background.close()
            finally:
                rgba.close()
    else:
        with budget.memory(image.width * image.height):
            budget.raster(*image.size)
            result = image.convert("RGB")
    image.close()
    return result


def _image_page(data: bytes, mime: str, budget: PixelBudget) -> ReferencePage:
    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = MAX_PAGE_PIXELS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=list(MIMES)) as probe:
                if MIMES.get(probe.format) != mime:
                    raise WorkerError("unsupported_media")
                dimensions(*probe.size)
                if getattr(probe, "n_frames", 1) != 1 or getattr(probe, "is_animated", False):
                    raise WorkerError("unsupported_media")
                # Verify compressed structure before allocating full pixel storage.
                probe.verify()
            with Image.open(io.BytesIO(data), formats=list(MIMES)) as original:
                with budget.memory(2 * original.width * original.height):
                    budget.raster(*original.size)
                    orientation = original.getexif().get(274, 1)
                    # Some Android screenshots explicitly store 0 rather than
                    # omitting unspecified orientation. Keep their stored pixel
                    # order, just as for an absent tag; never guess a rotation.
                    # ImageOps applies no transform for 0. All allocation/work
                    # charges and the immutable original bytes remain the same.
                    if type(orientation) is not int or orientation not in range(0, 9):
                        raise WorkerError("invalid_document")
                    original.load()
                    # exif_transpose also copies when orientation is already 1.
                    budget.raster(*original.size)
                    oriented = ImageOps.exif_transpose(original)
                    try:
                        rgb = _rgb(oriented, budget)
                    except BaseException:
                        oriented.close()
                        raise
            budget.hold(rgb.width * rgb.height)
            return ReferencePage(1, rgb.width, rgb.height, rgb, budget=budget)
    except WorkerError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise WorkerError("resource_limit", resource_limit="page_dimensions") from None
    except MemoryError:
        raise WorkerError("resource_limit", resource_limit="memory") from None
    except Exception:
        raise WorkerError("invalid_document") from None


def _pdf_pages(data: bytes, scale: float, budget: PixelBudget,
               selected_pages: frozenset[int] | None = None) -> Iterator[ReferencePage]:
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    try:
        document = pdfium.PdfDocument(data)
    except Exception:
        # v4.30's PdfiumError has no error-code attribute; consult PDFium directly.
        if raw.FPDF_GetLastError() == raw.FPDF_ERR_PASSWORD:
            raise WorkerError("encrypted_document") from None
        raise WorkerError("invalid_document") from None
    try:
        # Also reject encrypted PDFs that happen to accept an empty password.
        if raw.FPDF_GetSecurityHandlerRevision(document) != -1:
            raise WorkerError("encrypted_document")
        # Dynamic form appearance needs a separately validated renderer; do not
        # silently omit fields or initialize JavaScript/XFA form execution.
        if document.get_formtype() != raw.FORMTYPE_NONE:
            raise WorkerError("unsupported_media")
        count = len(document)
        if count < 1:
            raise WorkerError("invalid_document")
        if count > MAX_PAGES:
            raise WorkerError("resource_limit", resource_limit="page_count")
        if selected_pages is not None and any(number > count for number in selected_pages):
            raise WorkerError("invalid_document")
        sizes: list[tuple[int, int]] = []
        # Preflight every page before allocating any raster or calling a model.
        for index in range(count):
            page = document[index]
            try:
                # PDFium width/height/render already apply inherited CropBox and
                # /Rotate. Do not apply get_rotation() a second time.
                sizes.append(dimensions(page.get_width() * scale, page.get_height() * scale))
                bbox = page.get_bbox()
                if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    raise WorkerError("invalid_document")
            finally:
                page.close()
        if sum(width * height for width, height in sizes) > MAX_TOTAL_PIXELS:
            raise WorkerError("resource_limit", resource_limit="pixel_budget")
        for index, (width, height) in enumerate(sizes):
            if selected_pages is not None and index + 1 not in selected_pages:
                continue
            # Native bitmap, possible PIL frombuffer copy (RGB is not mapped),
            # and explicit detached copy. Reserve before rendering starts.
            work = 3 * width * height
            page = None
            bitmap = None
            raster = None
            rgb = None
            render_failed = False
            budget.hold(work)
            try:
                budget.reserve(work)
                page = document[index]
                bitmap = page.render(scale=scale, rotation=0, crop=(0, 0, 0, 0),
                    may_draw_forms=False, draw_annots=True, rev_byteorder=True,
                    fill_color=(255, 255, 255, 255), limit_image_cache=True)
                if (bitmap.width, bitmap.height) != (width, height):
                    raise WorkerError("invalid_document")
                # Detach from native bitmap before closing its backing memory.
                raster = bitmap.to_pil().copy()
                rgb = _rgb(raster, budget)
            except WorkerError:
                raise
            except MemoryError:
                raise WorkerError("resource_limit", resource_limit="memory") from None
            except Exception:
                render_failed = True
            finally:
                if raster is not None and rgb is None:
                    raster.close()
                if bitmap is not None:
                    bitmap.close()
                if page is not None:
                    page.close()
                budget.release(work)
            if render_failed:
                yield ReferencePage(index + 1, width, height, None, "The page could not be rendered.")
                continue
            budget.hold(width * height)
            reference = ReferencePage(index + 1, width, height, rgb, budget=budget)
            try:
                yield reference
            finally:
                reference.close()
    except WorkerError:
        raise
    except MemoryError:
        raise WorkerError("resource_limit", resource_limit="memory") from None
    except Exception:
        raise WorkerError("invalid_document") from None
    finally:
        document.close()


def decode_pages(data: bytes, mime: str, budget: PixelBudget, pdf_scale: float = 2.0,
                 *, selected_pages: frozenset[int] | None = None) -> Iterator[ReferencePage]:
    if not data or len(data) > MAX_INPUT_BYTES:
        raise WorkerError("resource_limit", resource_limit="input_bytes")
    if pdf_scale != 2:
        raise WorkerError("provider_unavailable")
    # An internal recovery pass may render an exact subset of the same original.
    # All original PDF safety/page/dimension checks still run before rendering.
    if selected_pages is not None and (
        not isinstance(selected_pages, frozenset) or not selected_pages
        or len(selected_pages) > MAX_PAGES
        or any(type(number) is not int or not 1 <= number <= MAX_PAGES for number in selected_pages)
    ):
        raise WorkerError("invalid_document")
    if mime == "application/pdf":
        yield from _pdf_pages(data, pdf_scale, budget, selected_pages)
    elif mime in MIMES.values():
        if selected_pages is not None and selected_pages != frozenset({1}):
            raise WorkerError("invalid_document")
        page = _image_page(data, mime, budget)
        try:
            yield page
        finally:
            page.close()
    else:
        raise WorkerError("unsupported_media")
