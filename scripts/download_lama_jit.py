"""
Download a TorchScript-compiled big-lama model to lama/big-lama/big-lama.pt.

This is the lightest possible way to run LaMa inside UNTORN — no
pytorch-lightning / kornia / albumentations / saicinpainting needed, just
torch. After running this script once, untorn.inpainting will pick up the
JIT model automatically on its next invocation.

Source release: https://github.com/enesmsahin/simple-lama-inpainting/releases/tag/v0.1.0
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/"
    "v0.1.0/big-lama.pt"
)
DEST = Path(r"C:\dev\untorn\lama\big-lama\big-lama.pt")
EXPECTED_MIN_BYTES = 190_000_000   # ~200 MB


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)

    if DEST.exists() and DEST.stat().st_size >= EXPECTED_MIN_BYTES:
        print(f"[✓] Already present: {DEST}  ({_human(DEST.stat().st_size)})")
        return 0

    print(f"Downloading big-lama TorchScript model ...")
    print(f"  from: {URL}")
    print(f"  to:   {DEST}")

    last_pct = -1

    def hook(block_num: int, block_size: int, total_size: int):
        nonlocal last_pct
        if total_size <= 0:
            return
        got = block_num * block_size
        pct = int(100 * got / total_size)
        if pct != last_pct and pct % 2 == 0:
            sys.stdout.write(
                f"\r  {_human(min(got, total_size))} / {_human(total_size)}  ({pct}%)"
            )
            sys.stdout.flush()
            last_pct = pct

    try:
        urllib.request.urlretrieve(URL, str(DEST), reporthook=hook)
    except Exception as exc:
        print(f"\n[!] Download failed: {exc}")
        if DEST.exists():
            DEST.unlink()
        return 1

    print()
    size = DEST.stat().st_size
    print(f"[✓] Downloaded {_human(size)} → {DEST}")

    if size < EXPECTED_MIN_BYTES:
        print(f"[!] File is smaller than expected ({_human(EXPECTED_MIN_BYTES)}+); possibly corrupt.")
        return 1

    sha = hashlib.sha256()
    with open(DEST, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    print(f"    sha256: {sha.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
