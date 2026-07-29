"""
Command-line interface for document parsing pipeline.

Supports single file or batch directory processing.
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

from onion_parsing.core.pipeline import Pipeline
from onion_parsing.core.logging import setup_logging


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf'}


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser with all CLI options"""
    p = argparse.ArgumentParser(
        prog="onion_parsing",
        description="Document parsing pipeline (single file or batch directory)"
    )
    
    p.add_argument(
        "-i", "--img_path",
        required=True,
        help="Input image file or directory path"
    )
    p.add_argument(
        "-o", "--output_path",
        required=True,
        help="Output root directory"
    )
    p.add_argument(
        "-c", "--config",
        help="YAML configuration file path"
    )
    p.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode (save crops, intermediate results, bbox markdown)"
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    p.add_argument(
        "--skip",
        help="Comma-separated list of processors to skip, e.g. reorder,ocr"
    )
    p.add_argument(
        "--only",
        help="Comma-separated list of processors to run, e.g. coarse_detector,ocr"
    )
    p.add_argument(
        "--ignore_labels",
        default="header,advert",
        help="Comma-separated labels to filter in big crop (default: header,advert)"
    )
    
    return p


def collect_tasks(src: Path) -> List[Path]:
    """Collect valid image files from path"""
    if src.is_file():
        return [src]
    if src.is_dir():
        files = [f for f in src.iterdir() if f.suffix.lower() in IMG_EXTS]
        return sorted(files)
    raise FileNotFoundError(f"Path not found: {src}")


def parse_filter_list(raw: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated filter string into list"""
    if not raw:
        return None
    items = [x.strip() for x in raw.split(",")]
    return [x for x in items if x]


def setup_pipeline_config(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Build runtime pipeline configuration"""
    skip = parse_filter_list(args.skip)
    only = parse_filter_list(args.only)
    
    if not skip and not only:
        return None
    
    cfg: Dict[str, Any] = {"pipeline": {}}
    if skip:
        cfg["pipeline"]["skip_stages"] = skip
    if only:
        cfg["pipeline"]["processors"] = only
    
    return cfg


def write_markdown(path: Path, content: str) -> None:
    """Write markdown content to file"""
    path.write_text(content, encoding="utf-8")


def handle_debug_mode(
    task_file: Path,
    result: Dict[str, Any],
    debug_dir: Path,
    logger: logging.Logger
) -> None:
    """Process and save debug artifacts"""
    debug_dir.mkdir(exist_ok=True)
    
    from onion_parsing.utils.visualization import (
        render_detection,
        render_crop_boxes,
        dump_debug,
    )

    crops = result.get("crops", [])
    img_names = result.get("img_names", [])
    native_results = result.get("native_results", [])
    secondary_labels = result.get("secondary_labels", [])
    bigcrop_boxes = result.get("bigcrop_boxes", [])
    secondary_bboxes = result.get("secondary_bboxes", [])
    predict_md = result.get("predict_md", {})

    layout_viz = None
    if bigcrop_boxes:
        layout_viz = render_detection(
            str(task_file),
            boxes=bigcrop_boxes,
            color="red",
            show_label=True,
        )

    secondary_viz = None
    if secondary_bboxes:
        secondary_viz = render_crop_boxes(
            str(task_file),
            boxes=secondary_bboxes,
            labels=secondary_labels,
            color="blue",
        )

    dump_debug(
        str(debug_dir),
        crops=crops,
        names=img_names,
        raw_images=native_results,
        layout_img=layout_viz,
        crop_img=secondary_viz,
    )
    logger.info(f"Saved debug artifacts to: {debug_dir}")
    
    if predict_md and img_names:
        for md_name in img_names:
            md_data = predict_md.get(md_name)
            if md_data is None:
                continue
            md_text = md_data.get("markdown_texts", "") if isinstance(md_data, dict) else str(md_data)
            if md_text.strip():
                md_path = debug_dir / md_name / f"{md_name}.md"
                md_path.parent.mkdir(exist_ok=True)
                write_markdown(md_path, md_text)
        logger.info(f"Saved per-block markdown to: {debug_dir}")


def process_single(
    pipeline: Pipeline,
    task_file: Path,
    output_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger
) -> bool:
    """Process a single image file"""
    try:
        logger.info(f"Processing: {task_file.name}")
        
        result = pipeline.process(
            input_path=str(task_file),
            output_path=str(output_dir)
        )
        
        if result is None:
            logger.warning(f"No result for: {task_file.name}")
            return False
        
        final_md = result.get("final_clean_markdown", "")
        md_with_bbox = result.get("final_clean_markdown_with_bbox", "")
        
        md_path = output_dir / f"{task_file.stem}.md"
        write_markdown(md_path, final_md)
        logger.info(f"Saved markdown: {md_path}")
        
        if args.debug and md_with_bbox:
            bbox_md_path = output_dir / f"{task_file.stem}_with_bbox.md"
            write_markdown(bbox_md_path, md_with_bbox)
            logger.info(f"Saved bbox markdown: {bbox_md_path}")
            
            debug_dir = output_dir / "debug"
            handle_debug_mode(task_file, result, debug_dir, logger)
        
        logger.info(f"Completed: {task_file.name}")
        return True
        
    except Exception as exc:
        logger.error(f"Failed {task_file.name}: {exc}")
        traceback.print_exc()
        return False


def main() -> None:
    """Main CLI entry point"""
    parser = build_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger = logging.getLogger("cli")
    
    logger.info(f"Input: {args.img_path}")
    logger.info(f"Output: {args.output_path}")
    
    runtime_config = setup_pipeline_config(args)
    
    pipeline = Pipeline(
        config_path=args.config,
        runtime_config=runtime_config
    )
    
    input_path = Path(args.img_path)
    output_root = Path(args.output_path)
    
    try:
        tasks = collect_tasks(input_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    if not tasks:
        logger.error(f"No valid images found: {input_path}")
        sys.exit(1)
    
    logger.info(f"Found {len(tasks)} task(s)")
    
    success = 0
    for task_file in tasks:
        output_dir = output_root / task_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if process_single(pipeline, task_file, output_dir, args, logger):
            success += 1
    
    logger.info(f"\nAll done. {success}/{len(tasks)} succeeded. Results in: {args.output_path}")


if __name__ == "__main__":
    main()
