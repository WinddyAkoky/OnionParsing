"""
Visualization helpers for drawing debug overlays on images.

The public surface (render_detection, render_crop_boxes, dump_debug) is kept
fully compatible; only the internal mechanics were rewritten so that box
drawing, clamping and file IO each live in exactly one place.
"""

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# --- rendering knobs ---------------------------------------------------------
_REF_EDGE = 1920
_FONT_FLOOR = 12
_FONT_CEIL = 40
_FONT_BASE = 16
_BADGE_PAD = 8
_BADGE_INSET_X = 4
_BADGE_INSET_Y = 2


def _scale_for_font(edge: float) -> int:
    """Map the longer image edge to a font size bounded to a sane range."""
    ratio = edge / _REF_EDGE
    return max(_FONT_FLOOR, min(_FONT_CEIL, int(_FONT_BASE * ratio)))


def _to_pil(source: Any) -> Image.Image:
    """Coerce a path-like or ndarray input into an RGB PIL image."""
    if isinstance(source, np.ndarray):
        return Image.fromarray(source).convert("RGB")
    return Image.open(str(source)).convert("RGB")


def _iter_boxes(boxes, with_meta: bool):
    """Yield (coords, label) pairs from either dict-style or list-style boxes."""
    for entry in boxes:
        if with_meta and isinstance(entry, Mapping):
            yield entry.get("coordinate") or [], entry.get("label") or ""
        else:
            yield entry or [], None


class _BoxPainter:
    """Reusable drawing state that paints rectangles with caption badges."""

    def __init__(self, image: Image.Image):
        self._canvas = image
        self._pen = ImageDraw.Draw(image)
        self._w, self._h = image.size
        self._face = ImageFont.load_default(size=_scale_for_font(max(self._w, self._h)))

    @staticmethod
    def _clamp(value: int, ceiling: int) -> int:
        """Pin a coordinate into the [0, ceiling] range."""
        return int(min(max(value, 0), ceiling))

    def _badge(self, caption: str, x0: int, y0: int, edge_color: str, text_color: str) -> None:
        box = self._pen.textbbox((x0, y0), caption, font=self._face)
        tw, th = box[2] - box[0], box[3] - box[1]
        top = max(0, y0 - th - _BADGE_PAD)
        right = min(self._w, x0 + tw + _BADGE_PAD)
        self._pen.rectangle([(x0, top), (right, y0)], fill=edge_color)
        self._pen.text(
            (x0 + _BADGE_INSET_X, top + _BADGE_INSET_Y),
            caption, fill=text_color, font=self._face,
        )

    def draw(self, coords: Sequence[int], caption: str, edge_color: str,
             text_color: str, stroke: int) -> None:
        x0, y0, x1, y1 = (self._clamp(c, self._w if i % 2 == 0 else self._h) for i, c in enumerate(coords))
        self._pen.rectangle([(x0, y0), (x1, y1)], outline=edge_color, width=stroke)
        self._badge(caption, x0, y0, edge_color, text_color)

    def finish(self) -> np.ndarray:
        return np.array(self._canvas)


def _stamp(img_src, boxes, *, index_from_one: bool, with_meta: bool,
           color: str, text_col: str, thickness: int, label_of=None) -> np.ndarray:
    """Shared renderer behind both detection and crop overlays."""
    painter = _BoxPainter(_to_pil(img_src))
    for idx, (coords, label) in enumerate(_iter_boxes(boxes, with_meta)):
        if len(coords) != 4:
            continue
        seq = idx + 1 if index_from_one else idx + 1
        caption = label_of(idx, seq, label) if label_of else str(seq)
        painter.draw(coords, caption, color, text_col, thickness)
    return painter.finish()


def render_detection(
    img_src,
    boxes,
    color: str = "red",
    text_col: str = "yellow",
    thickness: int = 8,
    show_label: bool = False,
) -> np.ndarray:
    """Overlay primary detection boxes on image."""
    def caption(i, seq, label):
        return f"{seq}:{label}" if show_label and label else str(seq)
    return _stamp(img_src, boxes, index_from_one=True, with_meta=True,
                  color=color, text_col=text_col, thickness=thickness, label_of=caption)


def render_crop_boxes(
    img_src,
    boxes,
    labels=None,
    color: str = "blue",
    text_col: str = "white",
    thickness: int = 8,
) -> np.ndarray:
    """Overlay secondary crop boxes on image."""
    labels = list(labels or [])
    def caption(i, seq, label):
        return f"{seq}:{labels[i]}" if i < len(labels) and labels[i] else str(seq)
    return _stamp(img_src, boxes, index_from_one=True, with_meta=False,
                  color=color, text_col=text_col, thickness=thickness, label_of=caption)


def dump_debug(
    target_dir: str,
    crops,
    names,
    raw_images=None,
    layout_img=None,
    crop_img=None,
) -> None:
    """Persist intermediate debug artifacts to disk.

    Output tree:
        target_dir/
        ├── coarse_detector.png
        ├── fine_detector.png
        ├── crop_1/
        │   ├── crop_1.png
        │   └── crop_1_processed.png
        └── ...
    """
    root = Path(target_dir)
    root.mkdir(parents=True, exist_ok=True)

    def _write(arr, dest: Path) -> None:
        if arr is not None:
            Image.fromarray(arr).save(dest)

    _write(layout_img, root / "coarse_detector.png")
    _write(crop_img, root / "fine_detector.png")

    if not crops or not names:
        return

    for slot, name in enumerate(names):
        if slot >= len(crops):
            continue
        record = crops[slot]
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)

        _write(record.get("original_array"), folder / f"{name}.png")
        _write(record.get("array"), folder / f"{name}_processed.png")

        if raw_images and slot < len(raw_images) and raw_images[slot] is not None:
            exporter = raw_images[slot]
            if hasattr(exporter, "save_to_img"):
                try:
                    exporter.save_to_img(str(folder))
                except (IndexError, AttributeError, ValueError) as err:
                    logger.warning("Visualization save failed (%s): %s", name, err)


# --- backward-compatible aliases for renamed helpers --------------------------
open_image = _to_pil


def adaptive_font(w: int, h: int) -> int:
    return _scale_for_font(max(w, h))


def clip(val: int, lo: int, hi: int) -> int:
    return int(min(max(val, lo), hi))
