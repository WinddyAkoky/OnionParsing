"""
Reorder subsystem.

Sequences positioned text fragments into reading order by blending geometric
neighbourhood checks with NSP/MLM continuity predictions.
"""

from .reorder import ReorderProcessor, execute_reorder
from .reorder_models import arrange_markdown

reorder_markdown = arrange_markdown

__all__ = [
    "ReorderProcessor",
    "execute_reorder",
    "reorder_markdown",
    "arrange_markdown",
]
