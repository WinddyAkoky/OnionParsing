"""
Model provisioning facade for the reorder subsystem.

Offers a lazily-initialised handle to the NSP/MLM continuity predictor and a
high-level Markdown reordering entry point.
"""

from typing import Any, Optional

from onion_parsing.core.logging import get_logger

_log = get_logger(__name__)


def _resolve_runtime(target: str) -> Any:
    """Translate a device label into a ``torch.device`` (autodetects cuda)."""
    import torch

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass

    if target:
        return torch.device(target)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelHub:
    """Repository that materialises a predictor on first use."""

    def __init__(
        self,
        model_path: str,
        kind: str = "nsp",
        device: Optional[Any] = None,
        maxlen: int = 256,
        label: str = "npu:0",
    ) -> None:
        self._path = model_path
        self._kind = kind
        self._runtime = device if device is not None else _resolve_runtime(label)
        self._cap = maxlen
        self._delegate = None

    @property
    def engine(self) -> Any:
        """Lazily build and return the underlying predictor."""
        if self._delegate is None:
            _log.info("preparing %s model from %s", self._kind.upper(), self._path)
            if self._kind.lower() != "nsp":
                raise ValueError(
                    f"unsupported model kind '{self._kind}'; only 'nsp' is supported"
                )
            from onion_parsing.processors.reorder.predictors import NSPPredictor

            self._delegate = NSPPredictor(self._path, self._runtime, self._cap)
            self._delegate.load_model()
            _log.info("%s model ready", self._kind.upper())
        return self._delegate


def arrange_markdown(
    md_text: str,
    model_path: str = "/path/to/nsp_model",
    model_type: str = "nsp",
    device_id: str = "npu:0",
    max_length: int = 256,
    left_th: int = 200,
    below_th: int = 100,
    left_bottom_th: int = 1000,
    nsp_th: float = 0.6,
    dist_scale: int = 500,
) -> str:
    """Reorder the fragments inside ``md_text`` by reading order.

    Args:
        md_text: Markdown carrying ``bbox:[...]`` markers.
        model_path: Checkpoint directory for the continuity model.
        model_type: Predictor family (``"nsp"`` is the only supported value).
        device_id: Torch device label.
        max_length: Maximum token sequence length fed to the model.
        left_th: Lateral neighbourhood tolerance.
        below_th: Vertical neighbourhood tolerance.
        left_bottom_th: Diagonal neighbourhood tolerance.
        nsp_th: Minimum continuity score required to link two fragments.
        dist_scale: Spatial falloff constant.

    Returns:
        Reordered Markdown (bbox markers removed).
    """
    from onion_parsing.processors.reorder.reorder import execute_reorder

    return execute_reorder(
        md_text,
        model_path=model_path,
        model_type=model_type,
        device_str=device_id,
        max_length=max_length,
        left_threshold=left_th,
        below_threshold=below_th,
        below_left_threshold=left_bottom_th,
        nsp_threshold=nsp_th,
        distance_scale=dist_scale,
    )
