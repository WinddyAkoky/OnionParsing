"""
Preprocessor Processor - 图像预处理（修正版）
"""

import numpy as np
import cv2
from typing import Any, Dict, Optional
from onion_parsing.core.base import BaseProcessor
from onion_parsing.core.registry import register_processor
from onion_parsing.core.logging import get_logger


def white_padding(img_array, bg_color=(255, 255, 255)):
    """Centre ``img_array`` on a square canvas filled with ``bg_color``.

    Greyscale or RGBA sources are first normalised to three channels (RGBA is
    blended over the background colour), then dropped into the middle of a
    square whose edge is 1.5x the longest side. No scaling is applied.
    """
    src = img_array
    dims = src.ndim
    bands = src.shape[2] if dims == 3 else 1

    if dims == 2 or bands == 1:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    elif bands == 4:
        rgb = src[..., :3].astype(np.float32)
        cover = src[..., 3:4].astype(np.float32) * (1.0 / 255.0)
        back = np.asarray(bg_color, np.float32)
        blended = back[None, None, :] * (1.0 - cover) + rgb * cover
        src = blended.astype(np.uint8)

    h = src.shape[0]
    w = src.shape[1]
    side = int(max(h, w) * 3 / 2)

    canvas = np.empty((side, side, 3), np.uint8)
    canvas[:] = bg_color
    top = (side - h) >> 1
    left = (side - w) >> 1
    canvas[slice(top, top + h), slice(left, left + w)] = src
    return canvas


PAD_CROP_RANGE = (0.1, 10)


@register_processor("preprocessor")
class Preprocessor(BaseProcessor):
    """图像预处理Processor

    对图像进行白边填充（根据长宽比判断）
    """

    def __init__(self, name: str = "preprocessor", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.pad_crop_range = self.config.get("pad_crop_range", PAD_CROP_RANGE)

    def process(self, context: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """图像预处理

        Args:
            context: PipelineContext
            data: {"crops": [{"array": np.ndarray, ...}, ...]}

        Returns:
            dict: {"crops": [{"array": np.ndarray, ...}, ...]} (已预处理)
        """
        self.logger = get_logger("preprocessor")
        self.logger.info("执行图像预处理（白边填充）")

        crops_data = data.get("crops", [])
        if not crops_data:
            self.logger.warning("无Crop需要预处理")
            return {"crops": []}

        processed_crops = []
        for crop in crops_data:
            img_array = crop.get("array")

            if img_array is None:
                continue

            height, width = img_array.shape[:2]
            aspect_ratio = width / height

            if self.pad_crop_range[0] <= aspect_ratio <= self.pad_crop_range[1]:
                processed_array = white_padding(img_array)
            else:
                processed_array = img_array.copy()

            processed_crops.append({
                "array": processed_array,
                "original_array": img_array.copy(),
                "coordinate": crop.get("coordinate"),
                "label": crop.get("label"),
                "name": crop.get("name")
            })

        self.logger.info(f"预处理完成: {len(processed_crops)} 张Crop")
        result = {"crops": processed_crops}
        for passthrough in (
            "img_path",
            "bigcrop_boxes",
            "page_width",
            "page_height",
            "secondary_bboxes",
            "secondary_labels",
        ):
            if passthrough in data:
                result[passthrough] = data[passthrough]
        return result
