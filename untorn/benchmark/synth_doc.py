"""
untorn.benchmark.synth_doc
==========================
Procedural clean-document generator for benchmark inputs.

Produces a rectangular RGB image that resembles a typed document page
(white paper, dark text, optional horizontal rules). Used as the
clean source fed to the tear generator.

The benefit over using real scans: we can vary layout, ink density,
paper tone, line spacing — and we never accidentally tear up an input
that was itself a torn-paper scan.
"""

from __future__ import annotations

import numpy as np
import cv2


_LOREM = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five "
    "dozen liquor jugs. How vexingly quick daft zebras jump! Sphinx of "
    "black quartz, judge my vow. Two driven jocks help fax my big quiz. "
    "Amazingly few discotheques provide jukeboxes. Mr. Jock, TV quiz PhD, "
    "bags few lynx. The five boxing wizards jump quickly. Waltz, bad "
    "nymph, for quick jigs vex. Heavy boxes perform quick waltzes and "
    "jigs. Crazy Fredrick bought many very exquisite opal jewels."
)


def synth_document(width: int = 900,
                   height: int = 1200,
                   paper_color: tuple = (246, 242, 230),
                   ink_color: tuple = (25, 25, 30),
                   font_scale: float = 0.55,
                   line_height: int = 32,
                   margin: int = 70,
                   add_header: bool = True,
                   add_rules: bool = False,
                   seed: int = 0) -> np.ndarray:
    """
    Render a synthetic document page using OpenCV's Hershey fonts.

    Returns an RGB uint8 image.
    """
    rng = np.random.default_rng(seed)

    img = np.full((height, width, 3), paper_color, dtype=np.uint8)

    # Gentle paper grain
    noise = rng.normal(0, 2.0, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    y = margin
    if add_header:
        cv2.putText(img, "SYNTHETIC DOCUMENT", (margin, y + 26),
                    font, 0.9, ink_color, 2, lineType=cv2.LINE_AA)
        y += 60
        cv2.line(img, (margin, y), (width - margin, y), ink_color, 2)
        y += 30

    # Wrap LOREM into lines that fit within the margin
    words = _LOREM.split()
    max_text_width = width - 2 * margin
    line_words = []
    lines = []
    for word in words:
        trial = " ".join(line_words + [word])
        (tw, _), _ = cv2.getTextSize(trial, font, font_scale, thickness)
        if tw > max_text_width and line_words:
            lines.append(" ".join(line_words))
            line_words = [word]
        else:
            line_words.append(word)
    if line_words:
        lines.append(" ".join(line_words))

    # Repeat lines so the page fills up (real documents aren't half-empty)
    while (y + len(lines) * line_height) < (height - margin):
        lines = lines + lines
    lines = lines[: max(1, (height - margin - y) // line_height)]

    # Draw lines; add horizontal rules every few lines if requested
    for k, line in enumerate(lines):
        cv2.putText(img, line, (margin, y + 18),
                    font, font_scale, ink_color, thickness, cv2.LINE_AA)
        if add_rules and (k % 4 == 3):
            cv2.line(img, (margin, y + line_height - 4),
                     (width - margin, y + line_height - 4),
                     (200, 195, 180), 1)
        y += line_height
        if y + line_height > height - margin:
            break

    # Subtle corner vignette so edges are not perfectly sharp (more realistic)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    cx, cy = width / 2, height / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    vignette = 1.0 - 0.07 * np.clip(
        (dist - min(cx, cy) * 0.75) / max(cx, cy), 0, 1)
    img = np.clip(img.astype(np.float32) * vignette[..., None],
                  0, 255).astype(np.uint8)

    return img
