"""
OnionParsing PipelineExecutor

Pipeline执行器，编排阶段执行和错误处理
"""

from typing import List, Optional, Dict, Any
from onion_parsing.core.context import PipelineContext
from onion_parsing.core.registry import ProcessorRegistry
from onion_parsing.core.exceptions import OPPipelineError
from onion_parsing.core.logging import get_logger


class PipelineExecutor:
    """Pipeline执行器

    编排Pipeline阶段执行，处理错误和跳过机制。
    Processor实例在初始化时创建一次，跨所有execute()调用复用，
    确保模型不会每张图片重新加载。
    """

    def __init__(
        self,
        stages: List[str],
        config: Optional[Dict[str, Any]] = None,
        skip_stages: Optional[List[str]] = None
    ):
        self.stages = stages
        self.config = config or {}
        self.skip_stages = skip_stages or []
        self.logger = get_logger("executor")
        self._processors: Dict[str, Any] = {}
        self._build_processors()

    def _build_processors(self) -> None:
        """一次性创建所有Processor实例，供后续execute()复用"""
        for stage_name in self.stages:
            processor_class = ProcessorRegistry.get(stage_name)
            if processor_class is None:
                raise OPPipelineError(
                    stage_name,
                    f"Processor未注册: {stage_name}"
                )
            processor_config = self.config.get(stage_name, {})
            self._processors[stage_name] = processor_class(
                name=stage_name, config=processor_config
            )

    def execute(self, context: PipelineContext) -> Any:
        """执行Pipeline

        Args:
            context: Pipeline上下文

        Returns:
            最终结果

        Raises:
            OPPipelineError: Pipeline执行错误
        """
        data = {"img_path": str(context.input_path)}

        for stage_name in self.stages:
            if stage_name in self.skip_stages:
                self.logger.info(f"跳过阶段: {stage_name}")
                continue

            context.set_current_stage(stage_name)

            try:
                processor = self._processors[stage_name]

                self.logger.info(f"执行阶段: {stage_name}")
                data = processor.process(context, data)

                context.set_stage_result(stage_name, data)

            except Exception as e:
                self.logger.error(f"阶段执行失败: {stage_name} - {str(e)}")
                raise OPPipelineError(stage_name, str(e))

        return data