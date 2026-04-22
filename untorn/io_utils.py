"""
untorn.io_utils
===============
Image loading, saving, and format normalization.
"""

import cv2
import numpy as np
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk, convert to uint8 RGB numpy array.
    Handles 16-bit TIFs, grayscale, BGRA, etc.
    """
    path = str(path)
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {path}")

    # Convert to uint8 if needed (e.g. 16-bit TIF)
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.integer):
            max_val = np.iinfo(image.dtype).max
            image = (image.astype(np.float64) / max_val * 255).astype(np.uint8)
        else:
            image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

    # Convert to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_image(image_rgb: np.ndarray, path: str):
    """Save an RGB numpy array as an image file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def save_mask(mask: np.ndarray, path: str):
    """Save a single-channel mask."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)