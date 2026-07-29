"""
Markdown <-> text-fragment conversion helpers.

Translates a Markdown string carrying ``bbox:[...]`` markers into a list of
positioned fragments, and renders such a list back into Markdown prose.
"""

import logging
import re
from typing import Iterator, List, Optional, Tuple

from onion_parsing.processors.reorder.block import Block

_log = logging.getLogger(__name__)

# Four unsigned integers separated by commas, e.g. ``12, 34, 56, 78``.
_COORD_RUN = r"\d+(?:,\s*\d+){3}"
# A geometry marker such as  bbox:[12, 34, 56, 78].
_MARKER = "bbox:\\[" + _COORD_RUN + "\\]"
# Capturing form so re.split keeps the delimiters between text runs.
_MARKER_SPLIT = re.compile("(" + _MARKER + ")")
# A marker lingering at the very tail of a fragment's text.
_TAIL_MARKER = re.compile(r"\s*" + _MARKER + r"\s*\Z")
# Recover the raw coordinate digits from a captured marker.
_DIGITS = re.compile(r"\d+")

_PARAGRAPH_GAP = "\n\n"
_DEFAULT_GEOM = [0, 0, 1000, 100]


def extract(content: str) -> List[Block]:
    """Parse ``content`` (Markdown with bbox markers) into fragments.

    Args:
        content: Markdown source text.

    Returns:
        Fragments parsed from the source; empty when ``content`` is falsy.
    """
    if not content:
        _log.warning("refusing to parse empty markdown")
        return []
    return _build_blocks(content)


def _iter_payloads(content: str) -> Iterator[Tuple[Optional[List[int]], str]]:
    """Walk ``content`` split on geometry markers.

    A marker's rectangle is bound to the text that PRECEDES it. Leftover text
    trailing the final marker is yielded with a ``None`` rectangle so the caller
    can fall back to a default box.
    """
    chunks = _MARKER_SPLIT.split(content)
    stride = 2
    for head in range(0, len(chunks), stride):
        body = chunks[head]
        marker = chunks[head + 1] if head + 1 < len(chunks) else None
        rect = [int(d) for d in _DIGITS.findall(marker)] if marker else None
        yield rect, body


def _build_blocks(content: str) -> List[Block]:
    """Turn raw markdown into one fragment per non-empty payload."""
    fragments: List[Block] = []
    produced = 0
    for position, (rect, raw) in enumerate(_iter_payloads(content)):
        payload = raw.strip()
        if not payload:
            continue
        geom = rect if rect else list(_DEFAULT_GEOM)
        fragments.append(Block(payload, geom, position, produced))
        produced += 1
    _log.info("decoded %d fragments from markdown", len(fragments))
    return fragments


def _heading_only(text: str) -> bool:
    """``True`` when ``text`` is a lone heading line with no body."""
    compact = (text or "").strip()
    return "\n" not in compact and compact[:2] == "##"


def _opens_with_block(text: str) -> bool:
    """``True`` when ``text`` begins with a heading or markup tag."""
    compact = (text or "").lstrip()
    return compact[:2] == "##" or compact[:4] == "<div"


def _strip_bbox(raw: str) -> str:
    """Drop a trailing geometry marker along with any surrounding whitespace."""
    return _TAIL_MARKER.sub("", raw).strip()


def _linked(block: Block) -> bool:
    """Whether ``block`` reports itself as connected to its predecessor."""
    return bool(getattr(block, "is_connected", False))


def render(blocks: List[Block]) -> str:
    """Render fragments back into Markdown (bbox markers stripped).

    A blank line is inserted ahead of the current fragment when the previous
    fragment was a lone heading, the current fragment opens with a heading or
    markup tag, or the current fragment is flagged as disconnected.
    """
    if not blocks:
        return ""

    texts = [_strip_bbox(block.text) for block in blocks]
    rebuilt = [texts[0]]

    for earlier, current, block in zip(texts, texts[1:], blocks[1:]):
        detached = not _linked(block)
        needs_gap = detached or _heading_only(earlier) or _opens_with_block(current)
        if needs_gap:
            rebuilt.append(_PARAGRAPH_GAP)
        rebuilt.append(current)

    return "".join(rebuilt)


parse_markdown_from_str = extract
blocks_to_markdown = render
is_pure_title = _heading_only
starts_with_title = _opens_with_block
