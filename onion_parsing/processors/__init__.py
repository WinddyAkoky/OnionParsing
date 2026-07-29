"""
OnionParsing Processors Package

导入所有Processor模块以触发注册
"""

# 导入所有Processor以触发@register_processor装饰器
from onion_parsing.processors.coarse_detector import CoarseDetector
from onion_parsing.processors.sorter import Sorter
from onion_parsing.processors.cropper import Cropper
from onion_parsing.processors.crop_filter import CropFilter
from onion_parsing.processors.fine_detector import FineDetector
from onion_parsing.processors.column_expander import ColumnExpander
from onion_parsing.processors.preprocessor import Preprocessor
from onion_parsing.processors.ocr import OCR
from onion_parsing.processors.postprocessor import Postprocessor
from onion_parsing.processors.reorder import ReorderProcessor
from onion_parsing.processors.aggregator import Aggregator

__all__ = [
    "CoarseDetector",
    "Sorter",
    "Cropper",
    "CropFilter",
    "FineDetector",
    "ColumnExpander",
    "Preprocessor",
    "OCR",
    "Postprocessor",
    "ReorderProcessor",
    "Aggregator",
]