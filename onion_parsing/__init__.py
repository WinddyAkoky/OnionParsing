"""
OnionParsing - Document parsing pipeline.

Package: onion_parsing
Env prefix: OP_
"""

__version__ = "0.1.0"
__author__ = "OnionParsing Team"

from onion_parsing.core.exceptions import (
    OPError,
    OPInputError,
    OPConfigError,
    OPModelError,
    OPPipelineError,
)
from onion_parsing.core.pipeline import Pipeline

__all__ = [
    "Pipeline",
    "OPError",
    "OPInputError",
    "OPConfigError",
    "OPModelError",
    "OPPipelineError",
]
