"""Bounded, source-independent extension of the official SDK layout interface."""
from __future__ import annotations

from copy import deepcopy
import math

MAX_NATIVE_REGIONS = 512
MAX_PLANNED_REGIONS = 128
MIN_IMAGE_AREA = 10_000  # 1% of the normalized 1000 x1000 page.
MIN_PAIRED_TABLE_AREA = 30_000


def valid_region(region: object) -> bool:
    if not isinstance(region, dict) or region.get("task_type") not in {"text", "table", "formula", "skip"}:
        return False
    if not isinstance(region.get("label"), str):
        return False
    box = region.get("bbox_2d")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1000 for v in box):
        return False
    if box[0] >= box[2] or box[1] >= box[3]:
        return False
    polygon = region.get("polygon")
    if polygon is not None:
        if not isinstance(polygon, (list, tuple)) or not 4 <= len(polygon) <= 512:
            return False
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return False
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1000 for v in point):
                return False
        area = sum(polygon[i][0]*polygon[(i+1) % len(polygon)][1]
                   - polygon[(i+1) % len(polygon)][0]*polygon[i][1] for i in range(len(polygon)))
        if area == 0:
            return False
    return True


def plan_regions(native: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Keep SDK regions; pair substantial tables, omit tiny non-text images.

    Nothing recognized, no reference text, and no expected financial value
    selects a task. Every retained polygon/box is unchanged. The descriptor
    list binds the SDK's contiguous raw-result indexes to actual task origins.
    """
    if not isinstance(native, list) or len(native) > MAX_NATIVE_REGIONS:
        return [], [], {"invalid": 1, "tiny_images": 0, "limited": True}
    planned, descriptors = [], []
    stats = {"invalid": 0, "tiny_images": 0, "limited": False}

    def add(region, task, origin, parent=None):
        copy = deepcopy(region)
        copy.update(index=len(planned), task_type=task)
        if origin != "native":
            copy["label"] = "text"
        planned.append(copy)
        descriptors.append({"index":copy["index"], "task":task, "origin":origin,
                            "pairedFrom":parent, "bbox_2d":deepcopy(copy["bbox_2d"]),
                            "polygon":deepcopy(copy.get("polygon"))})

    for region in native:
        if not valid_region(region):
            stats["invalid"] += 1
            continue
        if region["task_type"] == "skip":
            continue
        left, top, right, bottom = region["bbox_2d"]
        area = (right-left)*(bottom-top)
        if region["label"] in {"image", "header_image", "footer_image"} and area < MIN_IMAGE_AREA:
            stats["tiny_images"] += 1
            continue
        if len(planned) >= MAX_PLANNED_REGIONS:
            stats["limited"] = True
            break
        parent = len(planned)
        add(region, region["task_type"], "native")
        if region["task_type"] == "table" and area >= MIN_PAIRED_TABLE_AREA:
            if len(planned) >= MAX_PLANNED_REGIONS:
                stats["limited"] = True
                continue
            add(region, "text", "paired-table-text", parent)
    return planned, descriptors, stats


class CoverageLayout:
    """Uses the SDK's documented custom-layout-detector extension point."""
    def __init__(self, detector):
        self.detector = detector
        self.batch_size = detector.batch_size
        self.config = detector.config
        self.plans = []
        self.stats = []

    def start(self):
        self.detector.start()

    def stop(self):
        self.detector.stop()

    def process(self, images, **kwargs):
        self.plans, self.stats = [], []
        native, visualizations = self.detector.process(images, **kwargs)
        if not isinstance(native, list) or len(native) != len(images):
            raise ValueError("GLM layout returned an invalid page inventory")
        results = []
        for regions in native:
            plan, descriptors, stats = plan_regions(regions)
            results.append(plan);self.plans.append(descriptors);self.stats.append(stats)
        return results, visualizations
