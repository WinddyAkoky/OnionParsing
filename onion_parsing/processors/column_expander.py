"""
Column-gap widening for multi-column document crops.

A vertical ink projection is analysed to locate inter-column gutters; every
gutter is then widened with a cosine-shaded ramp sampled from the ink colour on
either flank, yielding a seamless expanded page region.
"""

import numpy as np
import cv2
from scipy.signal import find_peaks

from onion_parsing.core.base import BaseProcessor
from onion_parsing.core.logging import get_logger
from onion_parsing.core.registry import register_processor

# channel-count -> cv2 grayscale conversion code
_GRAY_CONVERT = {
    3: cv2.COLOR_BGR2GRAY,
    4: cv2.COLOR_BGRA2GRAY,
}

# keys forwarded untouched through the processor
_PASSTHROUGH = (
    "img_path",
    "bigcrop_boxes",
    "page_width",
    "page_height",
    "secondary_bboxes",
    "secondary_labels",
)


class ColumnSpaceExpander:
    """Stateful pipeline widening the gutters of a single image."""

    def __init__(self, extra=30, bin_th=128, depth_lim=0.6, min_vw=2, min_cw=3, smooth_sz=5, jitter=2):
        self._pad = extra
        self._bin_th = bin_th
        self._depth = depth_lim
        self._vw = min_vw
        self._cw = min_cw
        self._kern = smooth_sz
        self._jitter = jitter

    # ----------------------------------------------------- colour / masks
    @staticmethod
    def to_gray(picture):
        # Collapse ``picture`` to a single-channel luminance plane.
        rank = picture.ndim if picture is not None else 0
        if rank == 2:
            return picture
        if rank != 3:
            raise ValueError("image rank unsupported")
        chan = picture.shape[2]
        if chan == 1:
            return picture[..., 0]
        conv = _GRAY_CONVERT.get(chan)
        if conv is None:
            raise ValueError("image channels unsupported")
        return cv2.cvtColor(picture, conv)

    def _ink(self, picture):
        # Binary foreground map where ink pixels are 255.
        gray = self.to_gray(picture)
        return cv2.threshold(gray, self._bin_th, 255, cv2.THRESH_BINARY_INV)[1]

    @staticmethod
    def _solid(height, width, template, colour):
        # Build a flat sheet of ``colour`` shaped like ``template``.
        is_grey = template.ndim == 2
        scalar = not hasattr(colour, "__len__")
        if is_grey:
            value = int(colour) if scalar else int(colour[0])
            return np.full((height, width), value).astype(template.dtype)
        bands = template.shape[2]
        if bands == 1:
            value = int(colour) if scalar else int(colour[0])
            return np.full((height, width, 1), value).astype(template.dtype)
        sheet = np.zeros((height, width, bands)).astype(template.dtype)
        limit = 1 if scalar else min(bands, len(colour))
        for ch, val in enumerate(colour[:limit]):
            sheet[:, :, ch] = int(val)
        if bands == 4:
            sheet[..., -1] = 255
        return sheet

    # ----------------------------------------------------- sampling
    def _background_stats(self, picture, spans):
        # Mean / std of background pixels lying outside the given columns.
        ink = self._ink(picture)
        forbidden = np.zeros(picture.shape[1], bool)
        for lo, hi in spans:
            forbidden[lo:hi].fill(True)

        grey = picture.ndim == 2
        stacks = []
        for col in np.flatnonzero(~forbidden):
            clear = ink[:, col] != 255
            chunk = (picture[:, col] if grey else picture[:, col, :])[clear]
            if chunk.size:
                stacks.append(chunk if not grey else chunk.reshape(-1, 1))

        if stacks:
            samples = np.vstack(stacks)
        else:
            samples = picture[ink != 255]
            if grey:
                samples = samples.reshape(-1, 1)

        return samples.mean(axis=0), samples.std(axis=0)

    @staticmethod
    def _edge_colour(picture, binary, centre, window=5):
        # Average colour of background pixels straddling ``centre``.
        half = window // 2
        left = max(0, centre - half)
        right = min(picture.shape[1], centre + half + 1)
        strip = picture[:, left:right]
        keep = binary[:, left:right] < 1

        picked = strip[keep]
        source = picked if picked.size else strip

        if picture.ndim == 2:
            return np.asarray([float(np.mean(source))])
        flat = source.reshape(-1, picture.shape[2])
        return np.mean(flat, axis=0)

    # ----------------------------------------------------- synthesis
    def _ramp(self, height, width, template, left, right):
        # Vectorised cosine-shaded blend from ``left`` to ``right``.
        span = max(1, 0 if template.ndim == 2 else template.shape[2])
        start = np.asarray(left[:span]).astype(np.float32)
        stop = np.asarray(right[:span]).astype(np.float32)

        positions = np.arange(width, dtype=np.float32)
        denom = (width - 1) if width > 1 else 1
        weight = (1.0 + np.cos(np.pi * positions / denom)) * 0.5

        row = np.clip(weight[:, None] * start[None, :] + (1.0 - weight[:, None]) * stop[None, :], 0.0, 255.0)
        row = row.astype(template.dtype)

        if template.ndim == 2:
            sheet = np.broadcast_to(row[:, 0], (height, width)).copy()
        else:
            sheet = np.broadcast_to(row[None, :, :], (height, width, span)).copy()

        if self._jitter > 0:
            amp = self._jitter
            noise = np.random.randint(1 - amp, amp + 2, size=sheet.shape, dtype=np.int16)
            shifted = sheet.astype(np.int16) + noise
            sheet = np.clip(shifted, 0, 255)
            sheet = sheet.astype(np.uint8)
        return sheet

    # ----------------------------------------------------- detection
    def _locate_gutters(self, picture):
        # Projection-valley analysis returning column spans and valley x's.
        ink = self._ink(picture)
        profile = ink.sum(axis=0).astype(np.float32)
        width = profile.shape[0]
        height = picture.shape[0]

        if height / max(width, 1) < 0.1:
            return [], np.empty(0)
        if width < self._kern:
            return ([(0, width)] if width >= self._cw else []), np.empty(0)

        kernel = np.ones(self._kern, dtype=np.float32) / self._kern
        smoothed = np.convolve(profile, kernel, mode="same")
        ceiling = smoothed.max()

        valleys, meta = find_peaks(ceiling - smoothed, width=self._vw)
        if valleys.size == 0:
            return [(0, width)], np.empty(0)

        def dip(values, refs):
            return np.where(refs > 0, np.clip((refs - values) / refs, 0.0, 1.0), 0.0)

        left_dip = dip(smoothed[valleys], smoothed[meta["left_bases"]])
        right_dip = dip(smoothed[valleys], smoothed[meta["right_bases"]])
        chosen = valleys[(left_dip >= self._depth) & (right_dip >= self._depth)]

        if chosen.size == 0:
            return [(0, width)], np.empty(0)

        spans = []
        left = 0
        for edge in chosen.astype(int):
            if edge - left >= self._cw:
                spans.append((left, edge))
            left = edge
        if width - left >= self._cw:
            spans.append((left, width))
        return spans, chosen

    # ----------------------------------------------------- orchestration
    def widen(self, picture):
        # Insert a shaded gap between every detected column.
        if picture is None:
            raise ValueError("image is None")

        spans, _ = self._locate_gutters(picture)
        if len(spans) < 2:
            return picture.copy()

        height, width = picture.shape[:2]
        ink = self._ink(picture)
        segments = []
        prev_end = 0

        for idx, (start, end) in enumerate(spans):
            if start > prev_end:
                segments.append(picture[:, prev_end:start])
            if idx > 0:
                left_col = self._edge_colour(picture, ink, spans[idx - 1][1] - 3, 6)
                right_col = self._edge_colour(picture, ink, spans[idx][0] + 3, 6)
                segments.append(self._ramp(height, self._pad, picture, left_col, right_col))
            segments.append(picture[:, start:end])
            prev_end = end

        if prev_end < width:
            segments.append(picture[:, prev_end:])
        return np.concatenate(segments, axis=1)


@register_processor("column_expander")
class ColumnExpander(BaseProcessor):
    """Expands spacing between columns in multi-column document images."""

    def __init__(self, name="column_expander", config=None):
        super().__init__(name, config)
        self._bin_th = self.config.get("threshold", 100)
        self._depth = self.config.get("depth_threshold", 0.60)
        self._pad = self.config.get("extra_spacing", 5)
        self._vw = self.config.get("min_valley_width", 2)
        self._cw = self.config.get("min_col_width", 18)
        self._kern = self.config.get("smooth_kernel", 4)
        self._jitter = self.config.get("color_jitter", 2)

    def _tool(self):
        return ColumnSpaceExpander(
            extra=self._pad, bin_th=self._bin_th, depth_lim=self._depth,
            min_vw=self._vw, min_cw=self._cw, smooth_sz=self._kern, jitter=self._jitter,
        )

    def process(self, context, data):
        # Apply column spacing expansion to crops.
        self.logger = get_logger("column_expander")
        self.logger.info("Expanding column spacing, threshold=%d", self._bin_th)

        crops = data.get("crops", [])
        if not crops:
            self.logger.warning("No crops to process")
            return {"crops": []}

        labels = data.get("secondary_labels", [])
        if len(crops) != len(labels):
            labels = [c.get("label", "region") for c in crops]

        tool = self._tool()
        results = []
        for item, label in zip(crops, labels):
            array = item.get("array")
            if array is None:
                continue
            if label != "region":
                results.append(item)
                continue
            results.append({
                "array": tool.widen(array),
                "original_array": item.get("original_array"),
                "coordinate": item.get("coordinate"),
                "label": label,
                "name": item.get("name"),
            })

        region_count = labels.count("region")
        self.logger.info(
            "Column expansion done: %d region images processed, %d skipped",
            region_count, len(labels) - region_count,
        )

        payload = {"crops": results}
        for key in _PASSTHROUGH:
            if key in data:
                payload[key] = data[key]
        return payload
