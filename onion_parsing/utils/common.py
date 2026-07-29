"""
Common Utilities
"""

from pathlib import Path
from typing import Any, Dict


def load_file(file_path: str) -> bytes:
    """加载文件内容"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(path, "rb") as f:
        return f.read()


def save_file(file_path: str, content: str) -> None:
    """保存文件内容"""
    path = Path(file_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def ensure_dir(dir_path: str) -> None:
    """确保目录存在"""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)