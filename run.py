"""
Untorn — CLI entry point
=========================
Usage:
    python run.py <input_image> [-o output_path]

Examples:
    python run.py data/input/test0001.tif
    python run.py data/input/photo.jpg -o my_result.png
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from untorn.pipeline import run


def main():
    parser = argparse.ArgumentParser(
        description="Untorn: Reconstruct torn paper fragments into a clean document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input image (.tif/.tiff/.jpg/.png)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output filename (saved in data/output/)")
    args = parser.parse_args()

    result = run(args.input, args.output)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()