"""
FineDetector Processor - 细粒度版面检测
"""

import os
import tempfile
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple
from onion_parsing.core.base import BaseProcessor
from onion_parsing.core.registry import register_processor
from onion_parsing.core.logging import get_logger
from onion_parsing.core.exceptions import OPModelError
from onion_parsing.utils.crop_decider import decide_crop
from onion_parsing.processors.sorter import reading_order_scaled

@register_processor("fine_detector")
class FineDetector(BaseProcessor):
    """二次切割Processor

    使用PaddleOCR LayoutDetection模型进行细粒度版面检测
    """

    def __init__(self, name: str = "fine_detector", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.threshold = self.config.get("threshold", 0.1)
        self.model_name = self.config.get("model_name", "PP-DocLayout_plus-L")
        self.model_dir = self.config.get("model_dir", "/path/to/fine_detection_model")
        self.device = self.config.get("device", "npu:0")
        self.merge_mode = self.config.get("layout_merge_bboxes_mode", "large")
        self.area_ratio_threshold = self.config.get("area_ratio_threshold", 1.0 / 8.0)
        self.aspect_ratio_ranges = self.config.get("aspect_ratio_ranges", [(0.15, 0.71), (1.45, 6.8)])
        self.smallcrop_scale = self.config.get("smallcrop_scale", (2.0 / 10, 2.0 / 30))
        self._model = None

    def _load_model(self) -> Any:
        """加载细粒度版面检测模型"""
        try:
            from paddleocr import LayoutDetection
            self.logger.info(f"加载细粒度LayoutDetection模型: model_name={self.model_name}, model_dir={self.model_dir}")
            return LayoutDetection(
                model_name=self.model_name,
                model_dir=self.model_dir,
                threshold=self.threshold,
                layout_merge_bboxes_mode=self.merge_mode,
                device=self.device
            )
        except ImportError as e:
            raise OPModelError("PaddleOCR", f"PaddleOCR未安装: {e}")

    @property
    def model(self) -> Any:
        """延迟加载模型"""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _should_secondary_crop(self, img_array: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """判断是否需要二次切割"""
        height, width = img_array.shape[:2]
        crop_area = height * width
        aspect_ratio = width / height
        return decide_crop(crop_area, crop_area, aspect_ratio, self.area_ratio_threshold, tuple(self.aspect_ratio_ranges))

    def _secondary_crop_impl(self, img_array: np.ndarray) -> Optional[List[Dict]]:
        """执行二次版面检测"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_path = tmp.name

        try:
            Image.fromarray(img_array).save(temp_path, quality=95)
            output = self.model.predict(temp_path, batch_size=16, layout_nms=True)

            if not output:
                return None

            res = output[0].json.get("res", {})
            boxes_raw = res.get("boxes", res) if isinstance(res, dict) else res

            valid_boxes = []
            for box in boxes_raw:
                coord = box.get("coordinate", [])
                label = box.get("label", box.get("block_label", "region"))
                if len(coord) >= 4:
                    valid_boxes.append({
                        "coordinate": coord[:4],
                        "label": label
                    })

            if not valid_boxes:
                return None

            # XY-Cut sorting for small crops (matches MGSO sort_bbox_for_smallcrops)
            coordinates = np.array([b["coordinate"] for b in valid_boxes])
            labels = [b["label"] for b in valid_boxes]
            sorted_coords, sorted_indices = reading_order_scaled(coordinates, self.smallcrop_scale)
            return [
                {"coordinate": [int(c) for c in coord], "label": labels[sorted_indices[i]]}
                for i, coord in enumerate(sorted_coords)
            ]

        except Exception as e:
            self.logger.error(f"细粒度版面检测失败: {e}")
            return None
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def process(self, context: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """二次切割"""
        self.logger = get_logger("fine_detector")
        self.logger.info(f"执行细粒度版面检测，threshold={self.threshold}")

        crops_data = data.get("crops", [])
        if not crops_data:
            self.logger.warning("无Crop需要二次切割")
            return {"crops": [], "secondary_bboxes": [], "secondary_labels": []}

        page_width = data.get("page_width", 0)
        page_height = data.get("page_height", 0)

        final_crops = []
        all_secondary_bboxes = []
        all_secondary_labels = []

        for i, crop in enumerate(crops_data, start=1):
            img_array = crop.get("array")
            bbox = crop.get("coordinate")
            label = crop.get("label", "region")

            if img_array is None or bbox is None:
                continue

            secondary_boxes = self._secondary_crop_impl(img_array)

            if not secondary_boxes:
                self.logger.warning(f"crop_{i} 二次切割失败，保留原Crop")
                final_crops.append({
                    "array": img_array,
                    "coordinate": bbox,
                    "label": label,
                    "name": f"crop_{i}"
                })
                continue

            img_pil = Image.fromarray(img_array)
            crop_sub_items = []
            crop_sub_bboxes = []
            crop_sub_labels = []
            for j, small_box in enumerate(secondary_boxes, start=1):
                coord = small_box.get("coordinate", [])
                small_label = small_box.get("label", "region")

                if len(coord) < 4:
                    continue

                x1, y1, x2, y2 = coord[:4]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_pil.width, x2), min(img_pil.height, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                small_array = np.array(img_pil.crop((x1, y1, x2, y2)))
                crop_sub_items.append({
                    "array": small_array,
                    "coordinate": [bbox[0] + x1, bbox[1] + y1, bbox[0] + x2, bbox[1] + y2],
                    "label": small_label,
                    "name": f"crop_{i}.{j}"
                })
                crop_sub_bboxes.append([bbox[0] + x1, bbox[1] + y1, bbox[0] + x2, bbox[1] + y2])
                crop_sub_labels.append(small_label)

            if crop_sub_items:
                final_crops.extend(crop_sub_items)
                all_secondary_bboxes.extend(crop_sub_bboxes)
                all_secondary_labels.extend(crop_sub_labels)
            else:
                self.logger.warning(f"crop_{i} 二次切割子区域全部退化，保留原Crop")
                final_crops.append({
                    "array": img_array,
                    "coordinate": bbox,
                    "label": label,
                    "name": f"crop_{i}"
                })

        self.logger.info(f"细粒度版面检测完成: {len(final_crops)} 个Crop")
        result = {
            "crops": final_crops,
            "secondary_bboxes": all_secondary_bboxes,
            "secondary_labels": all_secondary_labels
        }
        if "img_path" in data:
            result["img_path"] = data["img_path"]
        if "bigcrop_boxes" in data:
            result["bigcrop_boxes"] = data["bigcrop_boxes"]
        if page_width and page_height:
            result["page_width"] = page_width
            result["page_height"] = page_height
        return result
