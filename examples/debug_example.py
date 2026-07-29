#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Quick debug example for OnionParsing.

Usage:
    python examples/debug_example.py --mode single --input test.png --output output/test.md
    python examples/debug_example.py --mode reorder
    python examples/debug_example.py --mode stages
"""

import sys
import os

pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from onion_parsing.core.pipeline import Pipeline
from onion_parsing.core.logging import setup_logging


def debug_single_file(input_path: str, output_path: str):
    """Debug single file processing."""
    setup_logging(level="DEBUG")

    runtime_config = {
        "coarse_detector": {
            "threshold": 0.25,
            "device": "npu:0"
        },
        "ocr": {
            "vl_rec_server_url": "http://localhost:8118/v1",
            "timeout": 120
        },
        "reorder": {
            "nsp_threshold": 0.6,
            "left_threshold": 200,
            "below_threshold": 100
        }
    }

    pipeline = Pipeline(runtime_config=runtime_config)
    result = pipeline.process(input_path, output_path)

    if result:
        print("=" * 60)
        print("Processing complete!")
        print("=" * 60)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Markdown length: {len(result.get('final_clean_markdown', ''))}")
        print(f"Cropped images:  {len(result.get('final_img_arrays', []))}")
        print("=" * 60)

        final_md = result.get("final_clean_markdown", "")
        if final_md:
            print("\nMarkdown preview (first 500 chars):")
            print("-" * 60)
            print(final_md[:500])
            print("-" * 60)

    return result


def debug_reorder_only():
    """Debug the Reorder module standalone."""
    from onion_parsing.processors.reorder import arrange_markdown

    md_with_bbox = """
## Title 1
bbox:[100,200,300,400]

Some text content,
bbox:[100,400,300,600]

## Title 2
bbox:[350,200,550,400]

Some other text content.
bbox:[350,400,550,600]
"""

    print("=" * 60)
    print("Testing Reorder module")
    print("=" * 60)
    print("Input Markdown (with bbox):")
    print(md_with_bbox)
    print("=" * 60)

    final_md = arrange_markdown(md_with_bbox)

    print("Output Markdown (without bbox):")
    print("-" * 60)
    print(final_md)
    print("-" * 60)

    return final_md


def debug_pipeline_stages():
    """Debug Pipeline with stages skipped."""
    setup_logging(level="DEBUG")

    pipeline = Pipeline(skip_stages=["reorder"])

    input_path = "test.png"
    output_path = "output/test_no_reorder.md"

    print("=" * 60)
    print("Debug Pipeline (skip Reorder)")
    print("=" * 60)

    result = pipeline.process(input_path, output_path)

    if result:
        print("Stage results:")
        print(f"  - Layout detection: {len(result.get('boxes', []))} regions")
        print(f"  - Cropped images:   {len(result.get('final_img_arrays', []))}")
        print(f"  - OCR results:      {len(result.get('native_results', []))}")
        print(f"  - Markdown:         {len(result.get('final_clean_markdown', ''))} chars")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OnionParsing debug example")
    parser.add_argument(
        "--mode",
        choices=["single", "reorder", "stages"],
        default="single",
        help="Debug mode: single=single file, reorder=Reorder only, stages=Pipeline stages"
    )
    parser.add_argument(
        "--input",
        default="test.png",
        help="Input file path"
    )
    parser.add_argument(
        "--output",
        default="output/test.md",
        help="Output file path"
    )

    args = parser.parse_args()

    if args.mode == "single":
        debug_single_file(args.input, args.output)
    elif args.mode == "reorder":
        debug_reorder_only()
    elif args.mode == "stages":
        debug_pipeline_stages()
