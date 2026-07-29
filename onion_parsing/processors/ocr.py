"""
OCR Processor - OCR识别（修正版，使用PaddleOCRVL.predict）
"""

import numpy as np
from typing import Any, Dict, List, Optional
from onion_parsing.core.base import BaseProcessor
from onion_parsing.core.registry import register_processor
from onion_parsing.core.logging import get_logger
from onion_parsing.core.exceptions import OPModelError


@register_processor("ocr")
class OCR(BaseProcessor):
    """OCR识别Processor
    
    使用PaddleOCRVL.predict批量处理，返回native_results
    """
    
    def __init__(self, name: str = "ocr", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.timeout = self.config.get("timeout", 120)
        self.vl_rec_backend = self.config.get("vl_rec_backend", "vllm-server")
        self.vl_rec_server_url = self.config.get("vl_rec_server_url", "http://localhost:8118/v1")
        self.pipeline_version = self.config.get("pipeline_version", "v1.6")
        self.vl_rec_max_concurrency = self.config.get("vl_rec_max_concurrency", 32)
        self.vl_rec_fallback_concurrency = self.config.get("vl_rec_fallback_concurrency", 8)
        self.layout_threshold = self.config.get(
            "layout_threshold", {23: 0.1, 21: 0.85, 22: 0.2, 14: 0.8}
        )
        self.layout_unclip_ratio = self.config.get("layout_unclip_ratio", [1.1, 1.1])
        self.layout_merge_bboxes_mode = self.config.get("layout_merge_bboxes_mode", "large")
        self.device = self.config.get("device", "npu:0")
        self.enable_mkldnn = self.config.get("enable_mkldnn", True)
        self.cpu_threads = self.config.get("cpu_threads", 8)
        self._pipeline_vllm = None
        self._pipeline = None
    
    def _load_pipeline_vllm(self) -> Any:
        """加载vLLM加速的PaddleOCRVL"""
        try:
            from paddleocr import PaddleOCRVL
            self.logger.info(f"加载PaddleOCRVL（{self.vl_rec_backend}）模型...")
            params = {
                "pipeline_version": self.pipeline_version,
                "vl_rec_backend": self.vl_rec_backend,
                "vl_rec_server_url": self.vl_rec_server_url,
                "use_layout_detection": True,
                "layout_nms": True,
                "layout_threshold": self.layout_threshold,
                "layout_unclip_ratio": self.layout_unclip_ratio,
                "layout_merge_bboxes_mode": self.layout_merge_bboxes_mode,
                "device": self.device,
                "enable_mkldnn": self.enable_mkldnn,
                "cpu_threads": self.cpu_threads,
            }
            return PaddleOCRVL(**params, vl_rec_max_concurrency=self.vl_rec_max_concurrency)
        except ImportError as e:
            raise OPModelError("PaddleOCR", f"PaddleOCR未安装: {e}")
    
    def _load_pipeline(self) -> Any:
        """加载普通PaddleOCRVL（兜底）"""
        try:
            from paddleocr import PaddleOCRVL
            self.logger.info("加载PaddleOCRVL（非加速）模型...")
            params = {
                "pipeline_version": self.pipeline_version,
                "use_layout_detection": True,
                "layout_nms": True,
                "layout_threshold": self.layout_threshold,
                "layout_unclip_ratio": self.layout_unclip_ratio,
                "layout_merge_bboxes_mode": self.layout_merge_bboxes_mode,
                "device": self.device,
                "enable_mkldnn": self.enable_mkldnn,
                "cpu_threads": self.cpu_threads,
            }
            return PaddleOCRVL(**params, vl_rec_max_concurrency=self.vl_rec_fallback_concurrency)
        except ImportError as e:
            raise OPModelError("PaddleOCR", f"PaddleOCR未安装: {e}")
    
    @property
    def pipeline_vllm(self) -> Any:
        """延迟加载vLLM模型"""
        if self._pipeline_vllm is None:
            self._pipeline_vllm = self._load_pipeline_vllm()
        return self._pipeline_vllm
    
    @property
    def pipeline(self) -> Any:
        """延迟加载兜底模型"""
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline
    
    def process(self, context: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """OCR识别（批量处理，返回native_results）
        
        Args:
            context: PipelineContext
            data: {"crops": [{"array": np.ndarray, "name": "...", ...}, ...]}
        
        Returns:
            dict: {"img_names": [...], "native_results": [...], "crops": [...]}
        """
        self.logger = get_logger("ocr")
        self.logger.info("执行OCR识别（批量处理）")
        
        crops_data = data.get("crops", [])
        if not crops_data:
            self.logger.warning("无Crop需要OCR")
            return {"img_names": [], "native_results": [], "crops": []}
        
        img_arrays = [crop.get("array") for crop in crops_data if crop.get("array") is not None]
        img_names = [crop.get("name", f"crop_{i}") for i, crop in enumerate(crops_data) if crop.get("array") is not None]
        
        fail_from = len(img_arrays)
        native_results = [None] * len(img_arrays)
        
        try:
            self.logger.info(f"批量解析 {len(img_arrays)} 张图片")
            output = self.pipeline_vllm.predict(input=img_arrays)
            
            if not output:
                self.logger.error("批量返回为空，切换至兜底")
                fail_from = 0
            else:
                for i, (img_name, result) in enumerate(zip(img_names, output)):
                    native_results[i] = result
                    if result is None:
                        self.logger.warning(f"{img_name} 返回为 None")
                        continue
                
                if len(output) < len(img_arrays):
                    self.logger.warning(f"返回数量({len(output)})少于输入数量({len(img_arrays)})")
                    fail_from = len(output)
                else:
                    self.logger.info(f"批量解析成功 {len(img_arrays)} 张图片")
        
        except Exception as e:
            self.logger.error(f"批量解析失败: {e}，切换至兜底", exc_info=True)
            fail_from = 0
        
        fail_img = [(i, img_arrays[i], img_names[i]) for i in range(fail_from, len(img_arrays))]
        
        if fail_img:
            self.logger.info(f"vLLM失败 {len(fail_img)} 项，启用PaddleOCR兜底")
            for i, img_array, img_name in fail_img:
                try:
                    output = self.pipeline.predict(input=img_array)
                    if output:
                        native_results[i] = output[0] if isinstance(output, list) else output
                except Exception as e:
                    self.logger.error(f"PaddleOCR失败: {img_name}, {e}")
        
        self.logger.info(f"OCR完成: {len(img_names)} 个区域")
        result = {"img_names": img_names, "native_results": native_results, "crops": crops_data}
        if "img_path" in data:
            result["img_path"] = data["img_path"]
        if "bigcrop_boxes" in data:
            result["bigcrop_boxes"] = data["bigcrop_boxes"]
        if "secondary_bboxes" in data:
            result["secondary_bboxes"] = data["secondary_bboxes"]
        if "secondary_labels" in data:
            result["secondary_labels"] = data["secondary_labels"]
        return result