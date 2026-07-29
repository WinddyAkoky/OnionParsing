"""
Aggregator Processor - 结果聚合（修正版）
"""

import os
from typing import Any, Dict, List, Optional
from onion_parsing.core.base import BaseProcessor
from onion_parsing.core.registry import register_processor
from onion_parsing.core.logging import get_logger


@register_processor("aggregator")
class Aggregator(BaseProcessor):
    """结果聚合Processor
    
    输出Markdown文件
    """
    
    def __init__(self, name: str = "aggregator", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
    
    def process(self, context: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """结果聚合

        Args:
            context: PipelineContext
            data: {"final_clean_markdown": "...", ...}

        Returns:
            dict: {"markdown": "...", "output_path": "..."}
        """
        self.logger = get_logger("aggregator")
        self.logger.info("执行结果聚合")

        final_clean_markdown = data.get("final_clean_markdown", "")

        if not final_clean_markdown:
            from onion_parsing.processors.postprocessor import strip_bbox_lines
            final_clean_markdown = strip_bbox_lines(
                data.get("final_clean_markdown_with_bbox", "")
            )

        output_path = context.output_path
        if output_path:
            if os.path.isdir(output_path):
                input_name = os.path.basename(context.input_path)
                file_stem = os.path.splitext(input_name)[0]
                output_path = os.path.join(output_path, f"{file_stem}.md")

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_clean_markdown)

            self.logger.info(f"Markdown已保存: {output_path}")

        self.logger.info(f"聚合完成: {len(final_clean_markdown)} 字符")
        result = {"final_clean_markdown": final_clean_markdown, "output_path": output_path}
        # pass through fields needed for debug output
        for key in ("crops", "img_names", "secondary_labels", "secondary_bboxes", "native_results",
                     "predict_md", "final_clean_markdown_with_bbox",
                     "layout_visualization", "secondary_visualization", "bigcrop_boxes"):
            if key in data:
                result[key] = data[key]
        return result