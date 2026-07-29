"""
OnionParsing异常体系

定义所有异常类，统一错误处理
"""

from typing import Dict, Any, Optional


class OPError(Exception):
    """OnionParsing基础异常类"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class OPInputError(OPError):
    """输入相关异常"""
    
    def __init__(self, input_path: str, reason: str):
        message = f"输入错误: {input_path} - {reason}"
        details = {"input_path": input_path, "reason": reason}
        super().__init__(message, details)


class OPConfigError(OPError):
    """配置相关异常"""
    
    def __init__(self, config_key: str, reason: str):
        message = f"配置错误: {config_key} - {reason}"
        details = {"config_key": config_key, "reason": reason}
        super().__init__(message, details)


class OPModelError(OPError):
    """模型相关异常"""
    
    def __init__(self, model_name: str, reason: str):
        message = f"模型错误: {model_name} - {reason}"
        details = {"model_name": model_name, "reason": reason}
        super().__init__(message, details)


class OPPipelineError(OPError):
    """Pipeline相关异常"""
    
    def __init__(self, stage_name: str, reason: str):
        message = f"Pipeline错误: {stage_name} - {reason}"
        details = {"stage_name": stage_name, "reason": reason}
        super().__init__(message, details)