"""
CropFilter Processor - 大Crop过滤
"""

import numpy as np
from typing import Any, Dict, List, Optional
from onion_parsing.core.base import BaseProcessor
from onion_parsing.core.registry import register_processor
from onion_parsing.core.logging import get_logger


@register_processor("crop_filter")
class CropFilter(BaseProcessor):
    """大Crop过滤Processor
    
    根据标签过滤不需要的大Crop
    """
    
    DEFAULT_IGNORE_LABELS = ["header", "advert"]

    def __init__(self, name: str = "crop_filter", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.ignore_labels = self.config.get("ignore_labels", self.DEFAULT_IGNORE_LABELS)
    
    def process(self, context: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """过滤大Crop
        
        Args:
            context: PipelineContext
            data: {"crops": [{"array": np.ndarray, "coordinate": [...], "label": "..."}, ...]}
        
        Returns:
            dict: {"crops": [{"array": np.ndarray, "coordinate": [...], "label": "..."}, ...]} (已过滤)
        """
        self.logger = get_logger("crop_filter")
        self.logger.info(f"执行大Crop过滤，ignore_labels={self.ignore_labels}")
        
        crops_data = data.get("crops", [])
        if not crops_data:
            self.logger.warning("无Crop需要过滤")
            return {"crops": []}
        
        filtered_crops = []
        filtered_boxes = []
        bigcrop_boxes = data.get("bigcrop_boxes", [])
        for i, crop in enumerate(crops_data):
            label = crop.get("label", "")

            if label in self.ignore_labels:
                self.logger.info(f"过滤Crop: label={label}, coordinate={crop['coordinate']}")
                continue

            filtered_crops.append(crop)
            if i < len(bigcrop_boxes):
                filtered_boxes.append(bigcrop_boxes[i])

        self.logger.info(f"过滤完成: 保留 {len(filtered_crops)} 个Crop")
        result = {"crops": filtered_crops}
        if "img_path" in data:
            result["img_path"] = data["img_path"]
        if "bigcrop_boxes" in data:
            result["bigcrop_boxes"] = filtered_boxes
        if "page_width" in data:
            result["page_width"] = data["page_width"]
        if "page_height" in data:
            result["page_height"] = data["page_height"]
        return result