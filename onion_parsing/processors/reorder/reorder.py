"""
Fine-grained reading-order processor.

Wires model provisioning together with fragment parsing and chain assembly to
deliver an end-to-end Markdown reordering service.
"""

import re
from typing import Any, Dict, List, Optional

from onion_parsing.core.base import BaseProcessor
from onion_parsing.core.logging import get_logger
from onion_parsing.core.registry import register_processor


_GEOMETRY_LINE_RE = re.compile(r"bbox:\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]")
_TRAILING_BBOX_RE = re.compile(r"\n*bbox:\[\d+,\s*\d+,\s*\d+,\s*\d+\]\s*$")
_DEFAULT_RECT = [0, 0, 1000, 100]
_BLANK_GAP = "\n\n"


def extract_geometry(line: str) -> Optional[List[int]]:
    """Pull ``[x1, y1, x2, y2]`` out of a ``bbox:[...]`` marker on ``line``."""
    hit = _GEOMETRY_LINE_RE.search(line)
    if hit is None:
        return None
    return [int(hit.group(idx)) for idx in range(1, 5)]


def parse_blocks(markdown: str) -> List:
    """Split bbox-annotated Markdown into positioned fragments.

    Text preceding a ``bbox:[...]`` marker is bound to that marker's
    rectangle; trailing text with no following marker receives a fallback box.
    """
    from onion_parsing.processors.reorder.block import Block

    produced: List = []
    pending: List[str] = []
    start_line = 0
    next_id = 0

    for pos, raw in enumerate(markdown.split("\n")):
        rect = extract_geometry(raw)
        if rect is None:
            pending.append(raw)
            continue

        if pending:
            body = "\n".join(pending).rstrip()
            if body:
                produced.append(Block(body, rect, start_line, next_id))
                next_id += 1
            pending = []
        start_line = pos + 1

    if pending:
        body = "\n".join(pending).rstrip()
        if body:
            produced.append(Block(body, list(_DEFAULT_RECT), start_line, next_id))

    return produced


def compose_markdown(blocks: List) -> str:
    """Stitch ordered fragments back into Markdown.

    A blank line separates the current fragment from its predecessor when the
    predecessor is a lone heading, the current fragment opens with a heading or
    markup tag, or the current fragment is flagged as disconnected.
    """
    if not blocks:
        return ""

    bodies = [_strip_trailing_bbox(block.text) for block in blocks]
    rendered: List[str] = [bodies[0]]

    for idx in range(1, len(bodies)):
        body = bodies[idx]
        precursor = bodies[idx - 1]
        joined = getattr(blocks[idx], "is_connected", False)
        precursor_heading = precursor.count("\n") == 0 and precursor.startswith("##")
        body_opens_block = body.startswith("##") or body.startswith("<div")
        if precursor_heading or body_opens_block or not joined:
            rendered.append(_BLANK_GAP)
        rendered.append(body)

    return "".join(rendered)


def _strip_trailing_bbox(raw: str) -> str:
    return _TRAILING_BBOX_RE.sub("", raw).strip()


class ModelHub:
    """Lazy provider for the NSP continuity predictor."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "nsp",
        device_str: str = "npu:0",
        max_length: int = 256,
    ) -> None:
        self._path = model_path
        self._kind = model_type
        self._device_label = device_str
        self._cap = max_length
        self._cached = None

    def _resolve_device(self):
        import torch

        try:
            import torch_npu  # noqa: F401  (registers the npu runtime when available)
        except ImportError:
            pass

        if self._device_label:
            return torch.device(self._device_label)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def obtain(self):
        """Return the predictor, loading it on first access."""
        if self._cached is not None:
            return self._cached

        log = get_logger(__name__)
        if self._kind.lower() != "nsp":
            raise ValueError(
                f"unsupported model kind '{self._kind}'; only 'nsp' is permitted"
            )

        from onion_parsing.processors.reorder.predictors import NSPPredictor

        log.info("loading NSP weights from %s", self._path)
        self._cached = NSPPredictor(self._path, self._resolve_device(), self._cap)
        self._cached.load_model()
        log.info("NSP weights available")
        return self._cached


@register_processor("reorder")
class ReorderProcessor(BaseProcessor):
    """Processor that re-sequences fragments by reading order."""

    def __init__(self, name: str = "reorder", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config)
        self._tuning = {
            "left_threshold": self.config.get("left_threshold", 200),
            "below_threshold": self.config.get("below_threshold", 100),
            "below_left_threshold": self.config.get("below_left_threshold", 1000),
            "nsp_threshold": self.config.get("nsp_threshold", 0.6),
            "distance_scale": self.config.get("distance_scale", 500),
            "model_path": self.config.get("model_path", "/path/to/nsp_model"),
            "model_type": self.config.get("model_type", "nsp"),
            "device": self.config.get("device", "npu:0"),
            "max_length": self.config.get("max_length", 256),
        }
        self._model_hub = None

    @property
    def model_hub(self) -> "ModelHub":
        """Lazy-initialize ModelHub once and cache for all subsequent calls."""
        if self._model_hub is None:
            tuning = self._tuning
            self._model_hub = ModelHub(
                tuning["model_path"],
                tuning["model_type"],
                tuning["device"],
                tuning["max_length"],
            )
        return self._model_hub

    def process(self, context: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run fine-grained reordering over the inbound Markdown payload."""
        log = get_logger("reorder")
        log.info("starting fine-grained reordering")

        markdown = data.get("final_clean_markdown_with_bbox", "")
        if not markdown:
            log.warning("no bbox-annotated markdown supplied for reordering")
            return {"final_clean_markdown": "", **data}

        try:
            predictor = self.model_hub.obtain()

            from onion_parsing.processors.reorder.engine import ChainBuilder

            engine = ChainBuilder(
                predictor,
                left_threshold=self._tuning["left_threshold"],
                below_threshold=self._tuning["below_threshold"],
                below_left_threshold=self._tuning["below_left_threshold"],
                nsp_threshold=self._tuning["nsp_threshold"],
                distance_scale=self._tuning["distance_scale"],
            )

            fragments = parse_blocks(markdown)
            if not fragments:
                return {"final_clean_markdown": markdown, **data}

            sequenced = engine.reorder(fragments)
            rebuilt = compose_markdown(sequenced)

            log.info("fine-grained reordering complete")
            return {"final_clean_markdown": rebuilt, **data}

        except Exception as exc:  # noqa: BLE001
            log.error("fine-grained reordering failed: %s", exc)
            from onion_parsing.processors.postprocessor import strip_bbox_lines as remove_bbox_lines
            return {"final_clean_markdown": remove_bbox_lines(markdown), **data}


def execute_reorder(
    markdown: str,
    model_path: str = "/path/to/nsp_model",
    model_type: str = "nsp",
    device_str: str = "npu:0",
    max_length: int = 256,
    left_threshold: int = 200,
    below_threshold: int = 100,
    below_left_threshold: int = 1000,
    nsp_threshold: float = 0.6,
    distance_scale: int = 500,
) -> str:
    """Convenience entry point: reorder bbox-annotated Markdown end-to-end."""
    predictor = ModelHub(model_path, model_type, device_str, max_length).obtain()

    from onion_parsing.processors.reorder.engine import ChainBuilder

    engine = ChainBuilder(
        predictor,
        left_threshold=left_threshold,
        below_threshold=below_threshold,
        below_left_threshold=below_left_threshold,
        nsp_threshold=nsp_threshold,
        distance_scale=distance_scale,
    )

    fragments = parse_blocks(markdown)
    if not fragments:
        return markdown

    sequenced = engine.reorder(fragments)
    return compose_markdown(sequenced)
