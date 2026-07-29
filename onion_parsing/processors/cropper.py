"""
Cropper Processor - 区域裁剪
"""

import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional
from onion_parsing.core.base import BaseProcessor
from onion_parsing.core.registry import register_processor
from onion_parsing.core.logging import get_logger
from onion_parsing.core.exceptions import OPInputError


@register_processor("cropper")
class Cropper(BaseProcessor):
    """裁剪Processor
    
    根据版面检测结果裁剪图片区域
    """
    
    def __init__(self, name: str = "cropper", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
    
    def process(self, context: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """裁剪版面区域
        
        Args:
            context: PipelineContext
            data: {"boxes": [{"coordinate": [...], "label": "..."}, ...]}
        
        Returns:
            dict: {"crops": [{"array": np.ndarray, "coordinate": [...], "label": "..."}, ...]}
        """
        self.logger = get_logger("cropper")
        self.logger.info("执行区域裁剪")
        
        img_path = context.input_path
        boxes_data = data.get("boxes", [])
        
        if not boxes_data:
            self.logger.warning("无版面区域需要裁剪")
            return {"crops": []}
        
        try:
            original_image = Image.open(img_path)
        except Exception as e:
            raise OPInputError(img_path, f"无法打开图片: {e}")
        
        crops_list = []
        for box in boxes_data:
            coordinate = box.get("coordinate", [])
            label = box.get("label", "region")
            
            if len(coordinate) < 4:
                continue
            
            x1, y1, x2, y2 = coordinate[:4]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_image.width, x2), min(original_image.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop_array = np.array(original_image.crop((x1, y1, x2, y2)))
            crops_list.append({
                "array": crop_array,
                "coordinate": coordinate[:4],
                "label": label
            })
        
        self.logger.info(f"裁剪完成: {len(crops_list)} 个区域")
        result = {"crops": crops_list}
        if "img_path" in data:
            result["img_path"] = data["img_path"]
        if "boxes" in data:
            result["bigcrop_boxes"] = data["boxes"]
        result["page_width"] = original_image.width
        result["page_height"] = original_image.height
        return result