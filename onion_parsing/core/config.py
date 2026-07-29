"""
OnionParsing配置管理

实现四级配置优先级：运行时参数 > 环境变量 > YAML配置 > 默认配置
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config: Dict[str, Any] = {}
        self._load_default_config()
        if config_path:
            self._load_yaml(config_path)
        self._load_env_vars()
    
    def _load_default_config(self) -> None:
        """加载默认配置"""
        default_path = Path(__file__).parent.parent / "config" / "default.yaml"
        if default_path.exists():
            self.config = self._load_yaml_file(default_path)
    
    def _load_yaml(self, config_path: str) -> None:
        """加载YAML配置文件"""
        path = Path(config_path)
        if not path.exists():
            return
        yaml_config = self._load_yaml_file(path)
        self.config = self._deep_merge(self.config, yaml_config)
    
    def _load_yaml_file(self, path: Path) -> Dict[str, Any]:
        """加载YAML文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _load_env_vars(self) -> None:
        """加载环境变量并映射到配置路径
        
        映射规则：OP_<PROCESSOR>_<PARAM> → processor.param
        例如：OP_COARSE_DETECTOR_THRESHOLD → coarse_detector.threshold
        """
        env_config: Dict[str, Any] = {}
        for key, value in os.environ.items():
            if key.startswith("OP_"):
                config_key = self._map_env_key_to_config_key(key)
                if config_key:
                    env_config[config_key] = self._parse_env_value(value)
        self.config = self._deep_merge(self.config, env_config)
    
    def _map_env_key_to_config_key(self, env_key: str) -> Optional[str]:
        """映射环境变量键到配置键
        
        OP_COARSE_DETECTOR_THRESHOLD → coarse_detector.threshold
        OP_REORDER_NSP_THRESHOLD → reorder.nsp_threshold
        """
        parts = env_key[3:].split("_")
        if len(parts) < 2:
            return None
        processor = parts[0].lower()
        param_parts = [p.lower() for p in parts[1:]]
        param = "_".join(param_parts)
        return f"{processor}.{param}"
    
    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值"""
        try:
            if value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
            elif "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
    
    def _deep_merge(self, base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并配置字典"""
        result = base_dict.copy()
        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（支持嵌套路径）
        
        例如：get("coarse_detector.threshold") → 0.25
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update(self, overrides: Dict[str, Any]) -> None:
        """运行时更新配置"""
        self.config = self._deep_merge(self.config, overrides)