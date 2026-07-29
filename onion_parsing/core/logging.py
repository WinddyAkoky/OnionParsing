"""
OnionParsing日志系统

统一日志格式、OP前缀、onion_parsing.log输出
"""

import logging
import sys
from pathlib import Path
from typing import Optional

_LOGGING_SETUP = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format: str = "[%(asctime)s] [%(levelname)s] [OP.%(name)s] %(message)s"
) -> None:
    """设置日志系统
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径（默认：onion_parsing.log）
        format: 日志格式
    """
    global _LOGGING_SETUP
    
    if _LOGGING_SETUP:
        return
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    root_logger = logging.getLogger("onion_parsing")
    root_logger.setLevel(log_level)
    
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    formatter = logging.Formatter(format)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    if log_file is None:
        log_file = "onion_parsing.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    _LOGGING_SETUP = True


def get_logger(name: str) -> logging.Logger:
    """获取logger实例
    
    Args:
        name: logger名称（会自动添加onion_parsing前缀）
    
    Returns:
        Logger实例
    """
    return logging.getLogger(f"onion_parsing.{name}")