"""
OnionParsing基类定义

定义所有组件基类，提供统一接口和生命周期管理
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging


class BaseComponent(ABC):
    """组件基类"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"onion_parsing.{name}")
    
    def setup(self, context: Any) -> None:
        """初始化组件"""
        pass
    
    def teardown(self) -> None:
        """清理组件"""
        pass


class BaseProcessor(BaseComponent):
    """Processor基类"""
    
    @abstractmethod
    def process(self, context: Any, data: Any) -> Any:
        """处理数据"""
        raise NotImplementedError("子类必须实现 process 方法")


class BaseModel(BaseComponent):
    """模型基类，支持延迟加载"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._model: Optional[Any] = None
    
    @property
    def model(self) -> Any:
        """延迟加载模型"""
        if self._model is None:
            self.logger.info(f"延迟加载模型: {self.name}")
            self._model = self._load_model()
        return self._model
    
    @abstractmethod
    def _load_model(self) -> Any:
        """加载模型（子类实现）"""
        raise NotImplementedError("子类必须实现 _load_model 方法")
    
    def warm_up(self) -> None:
        """预热模型"""
        _ = self.model
        self.logger.info(f"模型预热完成: {self.name}")