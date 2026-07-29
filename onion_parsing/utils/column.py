"""
Column Detection Utility
"""

import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple


def detect_column_spacing(
    image_width: int,
    histogram: np.ndarray,
    threshold: int = 100,
    depth_threshold: float = 0.60,
    extra_spacing: int = 5
) -> List[Tuple[int, int]]:
    """检测列间距
    
    Args:
        image_width: 图像宽度
        histogram: 列直方图
        threshold: 峰值阈值
        depth_threshold: 深度阈值
        extra_spacing: 额外间距
    
    Returns:
        列间距列表 [(start, end), ...]
    """
    peaks, properties = find_peaks(histogram, height=threshold)
    
    if len(peaks) == 0:
        return []
    
    columns = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        gap_width = end - start
        
        if gap_width > threshold:
            columns.append((start + extra_spacing, end - extra_spacing))
    
    return columns