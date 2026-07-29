"""
Crop Decider Utility
"""

from typing import Dict, Any, Tuple


def decide_crop(
    crop_area: float,
    page_area: float,
    aspect_ratio: float,
    area_threshold: float = 0.125,
    aspect_ranges: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.15, 0.71), (1.45, 6.8))
) -> bool:
    """判断是否需要二次裁剪
    
    Args:
        crop_area: Crop面积
        page_area: 页面面积
        aspect_ratio: 长宽比
        area_threshold: 面积占比阈值（默认1/8）
        aspect_ranges: 长宽比范围
    
    Returns:
        是否需要二次裁剪
    """
    area_ratio = crop_area / page_area
    if area_ratio > area_threshold:
        return True
    
    for min_ratio, max_ratio in aspect_ranges:
        if min_ratio <= aspect_ratio <= max_ratio:
            return True
    
    return False