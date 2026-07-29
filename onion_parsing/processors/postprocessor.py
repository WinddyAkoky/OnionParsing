"""
Post-processor for OCR results.

Handles markdown extraction, title formatting, and crop merging.
"""

import re
from typing import Any, Dict, List, Optional

from onion_parsing.core.base import BaseProcessor
from onion_parsing.core.registry import register_processor
from onion_parsing.core.logging import get_logger


_NL_RAW = r"\n{3,}"
_PAREN_RAW = r'(\([^)]+\)\s*){5,}'
_DELIM_RAW = r'^[-=*_○×☐]{3,}\s*$'

RE_MULTI_NL = re.compile(_NL_RAW)
RE_MULTI_PAREN = re.compile(_PAREN_RAW)
RE_MULTI_DELIM = re.compile(_DELIM_RAW, re.MULTILINE)

BLOCK_STARTS = ("#", "<div", "<table", "<img", "<tr")
TRAIL_CHARS = "。！？)>|\";.；"
LEAD_CHARS = "【(▲"
TITLE_TAG = "title"


def strip_bbox_lines(md):
    # Drop every annotation line that begins with "bbox:".
    survivors = []
    for raw in md.split("\n"):
        head = raw.lstrip()
        if head[:5] != "bbox:":
            survivors.append(raw)
    return "\n".join(survivors)


def is_block_element(s):
    # True when the leading text is a markdown block element.
    return s.lstrip().startswith(BLOCK_STARTS)


def normalize_markdown(text):
    # Collapse stray blank lines and join lines that belong together.
    if not text:
        return text
    rows = text.split("\n")
    cleaned = []
    for ln in rows:
        stripped = ln.strip()
        if stripped:
            cleaned.append(stripped)
    if not cleaned:
        return ""
    merged = [cleaned[0]]
    for nxt in cleaned[1:]:
        last = merged[-1].rstrip()
        join_inline = (
            bool(last)
            and bool(nxt)
            and last[-1] not in TRAIL_CHARS
            and nxt[0] not in LEAD_CHARS
            and not is_block_element(last)
            and not is_block_element(nxt)
        )
        if join_inline:
            merged[-1] = last + nxt
        else:
            merged.append("\n\n" + nxt)
    joined = "\n".join(merged)
    return RE_MULTI_NL.sub("\n\n", joined)


def normalize_repeated_patterns(text):
    # Scrub excessive repeated brackets, delimiter rows and duplicated runs.
    if not text:
        return text
    for rgx in (RE_MULTI_PAREN, RE_MULTI_DELIM):
        text = rgx.sub("", text)
    text = collapse_duplicates(text)
    text = RE_MULTI_NL.sub("\n\n", text)
    cleaned = text.strip()
    return cleaned


def collapse_duplicates(text, min_rep=3):
    # Reduce a string that is entirely repetitions of one prefix pattern.
    if len(text) <= 1:
        return text
    distinct = set(text)
    if len(text) >= min_rep and len(distinct) == 1:
        return text[:1]
    cap = len(text) // min_rep
    for plen in range(1, cap + 1):
        unit = text[:plen]
        cycles = len(text) // plen
        if text == unit * cycles and cycles >= min_rep:
            return unit
    return "\n".join(shrink_line(line, min_rep) for line in text.split("\n"))


def shrink_line(line, min_rep=3):
    # Reduce repeated patterns confined to a single line.
    if len(line) <= 1:
        return line
    distinct = set(line)
    if len(line) >= min_rep and len(distinct) == 1:
        return line[:1]
    cap = len(line) // min_rep
    for plen in range(1, cap + 1):
        unit = line[:plen]
        cycles = len(line) // plen
        if line == unit * cycles and cycles >= min_rep:
            return unit
    return line


def pull_markdown(img_keys: List[str], native: List[Any]) -> Dict[str, str]:
    """Extract markdown text from native OCR results"""
    import logging
    from onion_parsing.utils.text_direction import judge

    log = logging.getLogger("postprocessor")
    outputs = {}

    for key, result in zip(img_keys, native):
        if result is None:
            log.warning("OCR result is None for %s — will be skipped", key)
            continue

        for block in result.get("parsing_res_list", []):
            bbox = block.bbox
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if block.label in ("text", "paragraph_title", "doc_title") and block.content and w > h:
                block.content = judge.fix_text(block.content)

        md_text = result._to_markdown(pretty=True)
        if isinstance(md_text, str) and not md_text.strip():
            log.warning("_to_markdown returned empty for %s — will be missing from final md", key)
        outputs[key] = md_text

    return outputs


def format_titles(md, labs, tag=TITLE_TAG):
    # Prepend "##" to the first line of every entry flagged as a title.
    if not labs:
        return md

    for idx in range(len(labs)):
        lab = labs[idx]
        if lab != tag:
            continue

        keys = list(md)
        if idx >= len(keys):
            continue

        key = keys[idx]
        content = md.get(key)
        if content is None:
            continue

        text = content.get("markdown_texts", "") if isinstance(content, dict) else str(content)
        if not text:
            continue

        parts = text.split("\n", 1)
        head = parts[0]
        if head[:2] == "# ":
            head = "#" + head
        elif head[:2] != "##":
            head = "## " + head

        merged = head
        if len(parts) > 1:
            tail = " ".join(parts[1].split("\n"))
            merged = head + " " + tail

        if isinstance(content, dict):
            md[key]["markdown_texts"] = merged
        else:
            md[key] = {"markdown_texts": merged}

    return md


def join_crops(
    md_data: Dict[str, Any],
    names: List[str],
    labs: List[str],
    bbox_map: Optional[Dict[int, tuple]] = None
) -> str:
    """Merge small crops grouped by big crop, using titles as separators"""
    pieces = []

    name_to_lab = dict(zip(names, labs))

    groups: Dict[str, List[tuple]] = {}
    order: List[str] = []

    for nm in names:
        data = md_data.get(nm)
        if data is None:
            continue

        text = data.get("markdown_texts", "") if isinstance(data, dict) else str(data)

        prefix = nm.rsplit(".", 1)[0]
        if prefix not in groups:
            order.append(prefix)
            groups[prefix] = []

        groups[prefix].append((text, name_to_lab.get(nm, "")))

    for prefix in order:
        items = groups[prefix]
        buffer: List[str] = []

        for txt, lab in items:
            if lab == TITLE_TAG:
                if buffer:
                    combined = "\n".join(buffer)
                    cleaned = normalize_markdown(normalize_repeated_patterns(combined))
                    if cleaned.strip():
                        pieces.append(cleaned)
                    buffer = []

                cleaned = normalize_markdown(normalize_repeated_patterns(txt))
                if cleaned.strip():
                    pieces.append(cleaned)
            else:
                buffer.append(txt)

        if buffer:
            combined = "\n".join(buffer)
            cleaned = normalize_markdown(normalize_repeated_patterns(combined))
            if cleaned.strip():
                pieces.append(cleaned)

        if bbox_map:
            try:
                idx = int(prefix.rsplit("_", 1)[-1])
                bbox = bbox_map.get(idx)
                if bbox:
                    pieces.append(f"bbox:[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]")
            except (ValueError, IndexError):
                pass

    return "\n\n".join(pieces)


@register_processor("postprocessor")
class Postprocessor(BaseProcessor):
    """Post-processes OCR results: extract markdown, format titles, merge crops"""

    def __init__(self, name: str = "postprocessor", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    def process(self, context: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process OCR results"""
        self.logger = get_logger("postprocessor")
        self.logger.info("Post-processing OCR results")

        img_names = data.get("img_names", [])
        native_results = data.get("native_results", [])
        crops = data.get("crops", [])
        secondary_labels = data.get("secondary_labels", [])

        if not native_results:
            self.logger.warning("No OCR results to process")
            out = {"predict_md": {}, "final_clean_markdown_with_bbox": "", "crops": crops, "secondary_labels": secondary_labels}
            if "secondary_bboxes" in data:
                out["secondary_bboxes"] = data["secondary_bboxes"]
            return out

        predict_md = pull_markdown(img_names, native_results)
        predict_md = format_titles(predict_md, secondary_labels, TITLE_TAG)

        bigcrop_boxes = data.get("bigcrop_boxes", [])
        bbox_index = {
            i: tuple(box.get("coordinate", []))
            for i, box in enumerate(bigcrop_boxes, start=1)
            if box.get("coordinate") and len(box.get("coordinate", [])) >= 4
        }

        final_md = join_crops(predict_md, img_names, secondary_labels, bbox_index)

        self.logger.info(f"Post-processing complete: {len(predict_md)} regions")
        return {
            "predict_md": predict_md,
            "final_clean_markdown_with_bbox": final_md,
            "native_results": native_results,
            "img_names": img_names,
            "crops": crops,
            "secondary_labels": secondary_labels,
            "img_path": data.get("img_path"),
            "bigcrop_boxes": data.get("bigcrop_boxes"),
            "secondary_bboxes": data.get("secondary_bboxes"),
        }
