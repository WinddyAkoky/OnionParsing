"""
CoarseDetector Processor - 粗粒度版面检测
"""
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional
from onion_parsing.core.base import BaseProcessor
from onion_parsing.core.registry import register_processor
from onion_parsing.core.logging import get_logger
from onion_parsing.core.exceptions import OPModelError


@register_processor("coarse_detector")
class CoarseDetector(BaseProcessor):
    """版面检测Processor

    使用PaddleOCR LayoutDetection模型（PP-DocLayout_plus-L）
    """

    def __init__(self, name: str = "coarse_detector", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.threshold = self.config.get("threshold", 0.25)
        self.target_size = self.config.get("target_size", (1240, 1755))
        self.model_name = self.config.get("model_name", "PP-DocLayout_plus-L")
        self.model_dir = self.config.get("model_dir", "/path/to/coarse_detection_model")
        self.device = self.config.get("device", "npu:0")
        self.merge_mode = self.config.get("layout_merge_bboxes_mode", "large")
        self._model = None

    def _load_model(self) -> Any:
        """加载LayoutDetection模型"""
        try:
            from paddleocr import LayoutDetection
            self.logger.info(f"加载LayoutDetection模型: model_name={self.model_name}, model_dir={self.model_dir}")
            return LayoutDetection(
                model_name=self.model_name,
                model_dir=self.model_dir,
                threshold=self.threshold,
                layout_merge_bboxes_mode=self.merge_mode,
                device=self.device
            )
        except ImportError as e:
            raise OPModelError("PaddleOCR", f"PaddleOCR未安装: {e}")

    @property
    def model(self) -> Any:
        """延迟加载模型"""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def process(self, context: Any, data: Any) -> Dict[str, Any]:
        """检测PDF版面布局

        Args:
            context: PipelineContext
            data: {"img_path": "..."} 或 None

        Returns:
            dict: {"boxes": [{"coordinate": [x1,y1,x2,y2], "label": "...", "confidence": 0.xx}, ...], "img_path": "..."}
        """
        self.logger = get_logger("coarse_detector")
        self.logger.info(f"执行版面检测，threshold={self.threshold}")

        img_path = data.get("img_path") if isinstance(data, dict) else str(context.input_path)

        original_image = Image.open(img_path)
        original_size = original_image.size
        need_resize = (original_size != self.target_size)

        if need_resize:
            resized_image = original_image.resize(self.target_size, Image.Resampling.LANCZOS)
            process_img = np.array(resized_image)
        else:
            process_img = np.array(original_image)

        try:
            output = self.model.predict(process_img, batch_size=1, layout_nms=True)
        except Exception as e:
            raise OPModelError("PaddleOCR", f"版面检测失败: {e}")

        boxes_list = []
        if output:
            res = output[0].json.get("res", {})
            boxes_raw = res.get("boxes", res) if isinstance(res, dict) else res

            for box in boxes_raw:
                coordinate = box.get("coordinate", [])
                label = box.get("label", "")
                confidence = box.get("score", box.get("confidence", 0.0))

                if len(coordinate) >= 4:
                    if need_resize:
                        scale_x = original_size[0] / self.target_size[0]
                        scale_y = original_size[1] / self.target_size[1]
                        coordinate = [
                            int(coordinate[0] * scale_x),
                            int(coordinate[1] * scale_y),
                            int(coordinate[2] * scale_x),
                            int(coordinate[3] * scale_y)
                        ]

                    boxes_list.append({
                        "coordinate": coordinate[:4],
                        "label": label,
                        "confidence": confidence
                    })

        self.logger.info(f"检测到 {len(boxes_list)} 个版面区域")
        return {"boxes": boxes_list, "img_path": img_path}
