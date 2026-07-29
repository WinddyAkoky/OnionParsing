"""
NSP Model
"""

from typing import Any
from onion_parsing.core.base import BaseModel
from onion_parsing.core.registry import register_model
from onion_parsing.core.logging import get_logger


@register_model("nsp")
class NSPModel(BaseModel):
    """NSP模型"""
    
    def _load_model(self) -> Any:
        """加载NSP模型"""
        self.logger = get_logger("nsp")
        self.logger.info("加载NSP模型")
        try:
            from transformers import BertForNextSentencePrediction
            model_name = self.config.get("model_name", "bert-base-chinese")
            return BertForNextSentencePrediction.from_pretrained(model_name)
        except ImportError:
            self.logger.error("transformers未安装")
            return None