"""
PaddleOCR Model
"""

from typing import Any
from onion_parsing.core.base import BaseModel
from onion_parsing.core.registry import register_model
from onion_parsing.core.logging import get_logger


@register_model("paddleocr")
class PaddleOCRModel(BaseModel):
    """PaddleOCR模型"""
    
    def _load_model(self) -> Any:
        """加载PaddleOCR模型"""
        self.logger = get_logger("paddleocr")
        self.logger.info("加载PaddleOCR模型")
        try:
            from paddleocr import PaddleOCR
            threshold = self.config.get("threshold", 0.25)
            return PaddleOCR(use_angle_cls=True, lang="ch", det_db_thresh=threshold)
        except ImportError:
            self.logger.error("PaddleOCR未安装")
            return None