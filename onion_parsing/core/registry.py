"""
OnionParsing Registry注册机制

实现Processor、Model、Pipeline的动态注册和查找
"""

from typing import Dict, Any, Type, Callable, Optional
from onion_parsing.core.base import BaseProcessor, BaseModel


class ProcessorRegistry:
    """Processor注册中心"""
    
    _registry: Dict[str, Type[BaseProcessor]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """注册装饰器
        
        Args:
            name: Processor名称
        
        Returns:
            装饰器函数
        """
        def decorator(processor_class: Type[BaseProcessor]) -> Type[BaseProcessor]:
            cls._registry[name] = processor_class
            return processor_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseProcessor]]:
        """获取已注册的Processor
        
        Args:
            name: Processor名称
        
        Returns:
            Processor类或None
        """
        return cls._registry.get(name)
    
    @classmethod
    def list(cls) -> Dict[str, Type[BaseProcessor]]:
        """列出所有已注册的Processor"""
        return cls._registry.copy()


class ModelRegistry:
    """Model注册中心"""
    
    _registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """注册装饰器
        
        Args:
            name: Model名称
        
        Returns:
            装饰器函数
        """
        def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
            cls._registry[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseModel]]:
        """获取已注册的Model
        
        Args:
            name: Model名称
        
        Returns:
            Model类或None
        """
        return cls._registry.get(name)
    
    @classmethod
    def list(cls) -> Dict[str, Type[BaseModel]]:
        """列出所有已注册的Model"""
        return cls._registry.copy()


class PipelineRegistry:
    """Pipeline注册中心"""
    
    _registry: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """注册装饰器
        
        Args:
            name: Pipeline名称
        
        Returns:
            装饰器函数
        """
        def decorator(pipeline_class: Any) -> Any:
            cls._registry[name] = pipeline_class
            return pipeline_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        """获取已注册的Pipeline
        
        Args:
            name: Pipeline名称
        
        Returns:
            Pipeline类或None
        """
        return cls._registry.get(name)
    
    @classmethod
    def list(cls) -> Dict[str, Any]:
        """列出所有已注册的Pipeline"""
        return cls._registry.copy()


register_processor = ProcessorRegistry.register
register_model = ModelRegistry.register
register_pipeline = PipelineRegistry.register