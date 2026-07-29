"""
OnionParsing PipelineContext

Pipeline上下文管理，维护状态和数据传递
"""

from typing import Any, Dict, Optional
from pathlib import Path


class PipelineContext:
    """Pipeline上下文
    
    维护Pipeline执行过程中的状态和数据
    """
    
    def __init__(self, input_path: str, output_path: str, config: Optional[Dict[str, Any]] = None):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.config = config or {}
        self._data: Dict[str, Any] = {}
        self._stage_results: Dict[str, Any] = {}
        self._current_stage: Optional[str] = None
    
    def set(self, key: str, value: Any) -> None:
        """设置上下文数据
        
        Args:
            key: 数据键
            value: 数据值
        """
        self._data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取上下文数据
        
        Args:
            key: 数据键
            default: 默认值
        
        Returns:
            数据值
        """
        return self._data.get(key, default)
    
    def set_stage_result(self, stage_name: str, result: Any) -> None:
        """设置阶段结果
        
        Args:
            stage_name: 阶段名称
            result: 阶段结果
        """
        self._stage_results[stage_name] = result
    
    def get_stage_result(self, stage_name: str, default: Any = None) -> Any:
        """获取阶段结果
        
        Args:
            stage_name: 阶段名称
            default: 默认值
        
        Returns:
            阶段结果
        """
        return self._stage_results.get(stage_name, default)
    
    def set_current_stage(self, stage_name: str) -> None:
        """设置当前执行阶段
        
        Args:
            stage_name: 阶段名称
        """
        self._current_stage = stage_name
    
    def get_current_stage(self) -> Optional[str]:
        """获取当前执行阶段
        
        Returns:
            当前阶段名称
        """
        return self._current_stage
    
    def get_all_stage_results(self) -> Dict[str, Any]:
        """获取所有阶段结果
        
        Returns:
            所有阶段结果字典
        """
        return self._stage_results.copy()