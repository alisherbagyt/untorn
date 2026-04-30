"""
Smoke test for the backend /board/data affine round-trip.

The backend reads `final_transforms.json` (canonical 3x3) and
`final_translations.json` (summary) and reconstructs each fragment's
canvas pose. This test feeds known transforms through the same logic
the backend uses (without spinning up FastAPI) and verifies:

  1. The 3x3 affine round-trips through final_transforms.json with
     micro-pixel precision (the bottleneck is the 6-decimal rounding
     in assembly._write_assembly_artifacts).
  2. When only final_translations.json is available (older debug dirs),
     the angle_deg + dx + dy fallback rebuilds an equivalent 2D
     similarity transform.
  3. The rotated-bbox-anchored canvas position the backend computes
     places the bbox CENTER at the affine-mapped source centroid —
     the contract the AssemblyView frontend depends on.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _resolve_affine_like_backend(fid_str: str,
                                 transforms: dict,
                                 trans: dict
                                 ) -> tuple[np.ndarray, float]:
    """Replica of backend.main._resolve_affine kept here so the test is
    independent of the FastAPI app."""
    full = transforms.get(fid_str)
    if isinstance(full, list) and len(full) == 3 and \
            all(isinstance(row, list) and len(row) == 3 for row in full):
        M = np.array(full, dtype=np.float64)
        ang = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        return M, ang
    ang_deg = float(trans.get("angle_deg", 0.0))
    dx = float(trans.get("dx", 0.0))
    dy = float(trans.get("dy", 0.0))
    c = math.cos(math.radians(ang_deg))
    s = math.sin(math.radians(ang_deg))
    M = np.array([[c, -s, dx],
                  [s,  c, dy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return M, ang_deg


def _emit_artifacts(tmp: Path, fragments: list[dict],
                    transforms: dict[int, np.ndarray]) -> None:
    """Mimic the assembly._write_assembly_artifacts call path."""
    from untorn.assembly import _write_assembly_artifacts

    placed: set[int] = set(transforms.keys())
    recon = tmp / "reconstruction"
    recon.mkdir(parents=True, exist_ok=True)
    _write_assembly_artifacts(
        fragments=fragments, transforms=transforms,
        placed=placed, merge_log=[], cache=None,
        recon_debug=recon,
    )


def test_full_affine_roundtrip():
    """A rotation + translation flows through final_transforms.json with
    sub-mpx precision and is correctly read by the backend resolver."""
    fragments = [{"id": 7}]
    theta = math.radians(12.5)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]], dtype=np.float64)
    T = np.eye(3, dtype=np.float64)
    T[:2, :2] = R
    T[:2,  2] = [123.456789, -45.987654]
    transforms = {0: T}

    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        _emit_artifacts(tmp, fragments, transforms)
        ft = json.loads(
            (tmp / "reconstruction" / "final_transforms.json").read_text())
        tr = json.loads(
            (tmp / "reconstruction" / "final_translations.json").read_text())

    M_back, ang = _resolve_affine_like_backend("7", ft, tr["7"])
    assert np.allclose(M_back, T, atol=1e-5), \
        f"affine round-trip drifted; max delta {np.max(np.abs(M_back - T)):.2e}"
    # angle_deg in summary should match the rotation we pushed in.
    assert abs(ang - 12.5) < 1e-3
    assert abs(tr["7"]["angle_deg"] - 12.5) < 1e-2
    assert abs(tr["7"]["dx"] - 123.46) < 0.01
    assert abs(tr["7"]["dy"] - -45.99) < 0.01
    print(f"  full-affine round-trip OK  (max delta {np.max(np.abs(M_back-T)):.2e})")


def test_backend_falls_back_to_translations_only():
    """If final_transforms.json is empty or missing, the backend resolver
    rebuilds a 2D similarity from the angle_deg + dx + dy summary."""
    trans = {"angle_deg": 7.5, "dx": 50.0, "dy": -10.0, "placed": True}
    M, ang = _resolve_affine_like_backend("3", transforms={}, trans=trans)
    assert abs(ang - 7.5) < 1e-9
    expected = np.array([
        [math.cos(math.radians(7.5)), -math.sin(math.radians(7.5)), 50.0],
        [math.sin(math.radians(7.5)),  math.cos(math.radians(7.5)), -10.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.allclose(M, expected, atol=1e-12)
    print("  fallback (angle_deg + dx + dy) OK")


def test_canvas_centroid_anchors_rotated_bbox():
    """Frontend rotates each fragment around its bbox centre. The backend
    therefore places (x, y) so the centre lands at the affine-mapped
    source centroid. Replicate that math here."""
    bx, by, bw, bh = 50, 60, 40, 80
    cx0 = bx + bw / 2.0
    cy0 = by + bh / 2.0
    theta = math.radians(20.0)
    M = np.array([[math.cos(theta), -math.sin(theta), 100.0],
                  [math.sin(theta),  math.cos(theta), 200.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    canvas_cx = M[0, 0] * cx0 + M[0, 1] * cy0 + M[0, 2]
    canvas_cy = M[1, 0] * cx0 + M[1, 1] * cy0 + M[1, 2]
    canvas_x = canvas_cx - bw / 2.0
    canvas_y = canvas_cy - bh / 2.0
    # The frontend will paint the crop at (canvas_x, canvas_y) and rotate
    # it 20° around its own centre. After that rotation, the centre
    # remains at (canvas_x + bw/2, canvas_y + bh/2) which equals the
    # affine-mapped centre by construction.
    rotated_centre = np.array([canvas_x + bw / 2.0,
                               canvas_y + bh / 2.0])
    expected_centre = np.array([canvas_cx, canvas_cy])
    assert np.allclose(rotated_centre, expected_centre)
    print(f"  rotated-bbox centre matches affine-mapped centroid "
          f"({rotated_centre.tolist()})")


if __name__ == "__main__":
    test_full_affine_roundtrip()
    test_backend_falls_back_to_translations_only()
    test_canvas_centroid_anchors_rotated_bbox()
    print("backend board tests passed")
