"""
OnionParsing Pipeline

Pipeline基类，整合Executor和Context
"""

from typing import Optional, Dict, Any, List
from onion_parsing.core.executor import PipelineExecutor
from onion_parsing.core.context import PipelineContext
from onion_parsing.core.config import ConfigManager
from onion_parsing.core.registry import PipelineRegistry
from onion_parsing.core.logging import setup_logging, get_logger

# 导入processors包以触发注册
import onion_parsing.processors


@PipelineRegistry.register("default")
class Pipeline:
    """Pipeline基类
    
    整合Executor、Context和ConfigManager
    """
    
    DEFAULT_STAGES = [
        "coarse_detector",
        "sorter",
        "cropper",
        "crop_filter",
        "fine_detector",
        "column_expander",
        "preprocessor",
        "ocr",
        "postprocessor",
        "reorder",
        "aggregator",
    ]
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        stages: Optional[List[str]] = None,
        skip_stages: Optional[List[str]] = None,
        runtime_config: Optional[Dict[str, Any]] = None
    ):
        self.config_manager = ConfigManager(config_path)
        
        if runtime_config:
            self.config_manager.update(runtime_config)
        
        self.stages = stages or self.config_manager.get(
            "pipeline.processors",
            self.DEFAULT_STAGES
        )
        
        self.skip_stages = skip_stages or []
        
        log_level = self.config_manager.get("logging.level", "INFO")
        log_file = self.config_manager.get("logging.file", "onion_parsing.log")
        setup_logging(level=log_level, log_file=log_file)
        
        self.logger = get_logger("pipeline")
        self.executor = PipelineExecutor(
            stages=self.stages,
            config=self.config_manager.config,
            skip_stages=self.skip_stages
        )
    
    def process(self, input_path: str, output_path: str) -> Any:
        """处理PDF文档
        
        Args:
            input_path: 输入PDF路径
            output_path: 输出Markdown路径
        
        Returns:
            解析结果
        """
        self.logger.info(f"开始处理: {input_path}")
        
        context = PipelineContext(
            input_path=input_path,
            output_path=output_path,
            config=self.config_manager.config
        )
        
        result = self.executor.execute(context)
        
        self.logger.info(f"处理完成: {output_path}")
        return result