"""
Sorter Processor - XY-Cut sorting
"""

import numpy as np
from typing import Any, Dict, Optional
from onion_parsing.core.base import BaseProcessor
from onion_parsing.core.registry import register_processor
from onion_parsing.core.logging import get_logger


def project(boxes, ax):
    """Build a per-axis projection histogram over the supplied boxes."""
    if ax not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    parts = boxes[:, ax::2]
    lo = parts.min()
    span = parts.max() if lo >= 0 else -lo
    acc = np.zeros(int(span), dtype=int)
    for seg in parts.tolist():
        acc[int(abs(seg[0])):int(abs(seg[1]))] += 1
    return acc


def extract_runs(vals, threshold, gap):
    """Split a projection profile into contiguous runs."""
    active = np.flatnonzero(vals > threshold)
    if active.size == 0:
        return None
    leaps = np.diff(active)
    cuts = np.flatnonzero(leaps > gap)
    head = active[cuts]
    tail = active[cuts + 1]
    starts = np.r_[active[0], tail]
    ends = np.r_[head, active[-1] + 1]
    return starts, ends


def xy_recursive(boxes, idx_arr, out, gap=1):
    """Recursive XY-cut that fills ``out`` with reading-order indices."""
    if len(boxes) != len(idx_arr):
        raise ValueError("boxes and indices length mismatch")
    order = np.argsort(boxes[:, 0])
    bx = boxes[order]
    ix = np.take(idx_arr, order)

    x_hist = project(bx, 0)
    x_runs = extract_runs(x_hist, threshold=0, gap=1)
    if x_runs is None:
        return

    xcol = bx[:, 0]
    xlo = np.abs(xcol)
    if xcol.min() < 0:
        x_runs = (np.flip(x_runs[0]), np.flip(x_runs[1]))

    for xs, xe in zip(x_runs[0], x_runs[1]):
        xm = (xlo >= xs) & (xlo < xe)
        chunk_b = bx[xm]
        chunk_i = ix[xm]

        order_y = np.argsort(chunk_b[:, 1])
        yb = chunk_b[order_y]
        yi = np.take(chunk_i, order_y)

        y_hist = project(yb, 1)
        y_runs = extract_runs(y_hist, 0, gap=gap)
        if y_runs is None:
            continue

        if y_runs[0].size == 1:
            out += yi.tolist()
            continue

        ylo = yb[:, 1]
        for ys, ye in zip(y_runs[0], y_runs[1]):
            ym = (ys <= ylo) & (ylo < ye)
            xy_recursive(yb[ym], yi[ym], out)


def scale_coords(coords, factors):
    """Apply a width/height scaling transform to every box."""
    sx, sy = factors
    span = np.max(np.abs(np.hstack((coords[:, 0], coords[:, 2]))))
    moved = coords.copy()
    moved[:, 0::2] += span
    out = np.empty_like(moved)
    for i, row in enumerate(moved):
        x1, y1, x2, y2 = row
        w = x1 - x2
        h = y2 - y1
        assert w > 0, "non-positive width"
        assert h > 0, "non-positive height"
        out[i] = [x2, y1, x1 - w * sx, y2 - h * sy]
    return out


def reading_order_scaled(coordinates, scale):
    """Order boxes by reading order, applying a scale transform first."""
    c = coordinates.copy()
    flip = np.array([-1, 1] * 2)
    c = scale_coords(c * flip, scale)

    ci = c.astype(int)
    positions = np.arange(c.shape[0])
    ordered = []
    xy_recursive(ci, positions, ordered)

    seen = set(ordered)
    missing = [k for k in range(c.shape[0]) if k not in seen]
    if missing:
        ordered.extend(missing)

    return coordinates[np.array(ordered)].tolist(), ordered


@register_processor("sorter")
class Sorter(BaseProcessor):
    """Sorter Processor - XY-Cut algorithm"""

    BIG_SCALE = (2.0 / 30, 20.0 / 30)

    def __init__(self, name: str = "sorter", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.scale = self.config.get("scale", self.BIG_SCALE)

    def process(self, context: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sort layout elements by reading order."""
        self.logger = get_logger("sorter")
        self.logger.info("Executing element sorting")

        boxes_data = data.get("boxes", [])
        if not boxes_data:
            self.logger.warning("No layout regions to sort")
            return data

        coords_list = []
        labels_list = []
        for box in boxes_data:
            coord = box.get("coordinate", [])
            label = box.get("label", "region")
            if len(coord) >= 4:
                coords_list.append([int(c) for c in coord[:4]])
                labels_list.append(label)

        if not coords_list:
            return data

        coords = np.array(coords_list)
        sorted_coords, sorted_indices = reading_order_scaled(coords, self.scale)

        sorted_boxes = []
        for i, coord in enumerate(sorted_coords):
            sorted_boxes.append({
                "coordinate": coord,
                "label": labels_list[sorted_indices[i]]
            })

        self.logger.info(f"Sorting complete: {len(sorted_boxes)} regions")
        result = {"boxes": sorted_boxes}
        if "img_path" in data:
            result["img_path"] = data["img_path"]
        return result
