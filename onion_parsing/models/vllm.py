"""
vLLM Model
"""

from typing import Any
from onion_parsing.core.base import BaseModel
from onion_parsing.core.registry import register_model
from onion_parsing.core.logging import get_logger


@register_model("vllm")
class vLLMModel(BaseModel):
    """vLLM模型"""
    
    def _load_model(self) -> Any:
        """加载vLLM模型"""
        self.logger = get_logger("vllm")
        self.logger.info("加载vLLM模型")
        try:
            from vllm import LLM
            model_name = self.config.get("model_name", "default")
            return LLM(model=model_name)
        except ImportError:
            self.logger.error("vLLM未安装")
            return None