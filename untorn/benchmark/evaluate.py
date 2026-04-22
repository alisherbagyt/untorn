"""
untorn.benchmark.evaluate
=========================
Score a pipeline output against ground truth.

Pipeline output lives in `data/debug/{case_id}/reconstruction/final_translations.json`
and the full final render at `data/output/{case_id}_reconstructed.png`.

The pipeline's coordinate frame is arbitrary — fragments are placed wherever
the greedy merge decides to put them, with the first-merged pair anchoring
the frame. So we never compare absolute positions. Instead:

    1. For every predicted fragment, compute its centroid in the pipeline's
       output canvas (T_pred @ centroid_in_input).
    2. Match predicted IDs to ground-truth IDs by nearest-centroid in the
       scan frame (pipeline INPUT = synthetic scan canvas).
    3. Compute the global similarity transform that best aligns predicted
       canvas centroids to ground-truth *source* centroids (Umeyama / 2D
       Procrustes). This absorbs the arbitrary framing of the pipeline's
       output canvas.
    4. Apply that transform and measure per-fragment residual error:
         - centroid translation error (px, in source frame)
         - rotation error (degrees) — the pipeline should INVERT the
           scatter rotation, so we compare pred_angle against -gt_angle.
    5. Also report: placement rate, pose-error-at-K percentiles, overall IoU.

Historical note: a previous version of this evaluator aligned predicted
canvas to the *scatter* canvas (GT's scan frame). That's wrong — no global
similarity can map a correctly-reconstructed document back to its scattered
scan layout, so metrics looked terrible even when the pipeline worked.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import cv2


@dataclass
class PerFragmentMetrics:
    frag_id: int
    centroid_error_px: float
    rotation_error_deg: float
    iou: float
    matched: bool


@dataclass
class CaseMetrics:
    case_id: str
    n_fragments_truth: int
    n_fragments_placed: int
    placement_rate: float
    median_centroid_error_px: float
    p90_centroid_error_px: float
    median_rotation_error_deg: float
    p90_rotation_error_deg: float
    mean_iou: float
    alignment_residual_px: float
    per_fragment: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ───────────────────────────────────────────────────────────────────────────
#  Core helpers
# ───────────────────────────────────────────────────────────────────────────

def _decompose_affine(M: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Decompose a 2x3 or 3x3 affine into (rotation_rad, translation_vec).
    Ignores scale/shear; paper is rigid so scale ≈ 1.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (3, 3):
        R = M[:2, :2]
        t = M[:2, 2]
    else:
        R = M[:, :2]
        t = M[:, 2]
    angle = math.atan2(R[1, 0], R[0, 0])
    return angle, t


def _umeyama_2d(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Find the 2D similarity transform (rotation + uniform scale + translation)
    that best maps src → dst in least-squares sense.
    Returns a 3x3 homogeneous matrix.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[1] == 2 and len(src) >= 2

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = (Vt.T @ D @ U.T)
    scale = (S @ np.diag(D)).sum() / (src_c ** 2).sum() \
        if (src_c ** 2).sum() > 1e-12 else 1.0
    t = mu_d - scale * R @ mu_s
    M = np.eye(3)
    M[:2, :2] = scale * R
    M[:2,  2] = t
    return M


def _apply_affine_pts(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts
    if M.shape == (3, 3):
        homog = np.hstack([pts, np.ones((len(pts), 1))])
        return (M @ homog.T).T[:, :2]
    return (M[:, :2] @ pts.T).T + M[:, 2]


def _rasterize_polygon_mask(poly: np.ndarray, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    return m > 0


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


# ───────────────────────────────────────────────────────────────────────────
#  Main evaluation
# ───────────────────────────────────────────────────────────────────────────

def evaluate_case(truth_path: str | Path,
                  predicted_transforms: dict,
                  predicted_fragment_centroids: dict,
                  predicted_fragment_polygons: Optional[dict] = None,
                  working_scale_factor: float = 1.0,
                  ) -> CaseMetrics:
    """
    Compute metrics for one case.

    Args:
        truth_path:                path to truth.json written by generate_case
        predicted_transforms:      dict {frag_id: 3x3 affine mapping the
                                   fragment's INPUT-space (working-res)
                                   coordinates to its predicted position on
                                   the pipeline OUTPUT canvas (also
                                   working-res).}
        predicted_fragment_centroids:
                                   dict {frag_id: (cx, cy)} centroid of each
                                   fragment in its segmented INPUT frame
                                   (working-res pixel coords).
        predicted_fragment_polygons:
                                   optional dict {frag_id: (N,2) polygon in
                                   its segmented frame} for IoU.
        working_scale_factor:      full / work ratio. GT lives in FULL-RES
                                   scan coords; predictions live in WORKING
                                   coords. Multiply predictions by this to
                                   compare in a common frame.

    We DO NOT assume the truth's fragment IDs match the pipeline's IDs.
    Instead we match predictions to GT by input-space centroid distance —
    each fragment occupies a unique region of the input, so centroid
    proximity is unambiguous.

    Returns a CaseMetrics instance.
    """
    truth_path = Path(truth_path)
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    gt_frags = truth["fragments"]

    s = float(working_scale_factor)

    # ── GT fragment centroids in BOTH canvas-frame (for matching) and
    #     source-frame (for alignment / error). ────────────────────────
    gt_centroids_canvas = []
    gt_centroids_source = []
    for g in gt_frags:
        M = np.array(g["affine_source_to_canvas"], dtype=np.float64)
        sc = np.array(g["centroid_in_source"], dtype=np.float64)
        cc = _apply_affine_pts(np.vstack([M, [0, 0, 1]]),
                               sc.reshape(1, 2))[0]
        gt_centroids_canvas.append(cc)
        gt_centroids_source.append(sc)
    gt_centroids_canvas = np.asarray(gt_centroids_canvas)
    gt_centroids_source = np.asarray(gt_centroids_source)

    # Only consider fragments that have BOTH a centroid and a transform.
    # The pipeline can bail on reconstruction (e.g. 0 valid matches), in
    # which case `final_translations.json` may be incomplete while
    # `fragments_meta.json` still lists every segmented fragment. Without
    # this guard we'd hit `KeyError: <pi>` below at
    #   T = predicted_transforms[pi]
    pred_ids = sorted(
        fid for fid in predicted_fragment_centroids.keys()
        if fid in predicted_transforms
    )
    if not pred_ids:
        return CaseMetrics(
            case_id=truth["case_id"],
            n_fragments_truth=len(gt_frags),
            n_fragments_placed=0,
            placement_rate=0.0,
            median_centroid_error_px=float("inf"),
            p90_centroid_error_px=float("inf"),
            median_rotation_error_deg=float("inf"),
            p90_rotation_error_deg=float("inf"),
            mean_iou=0.0,
            alignment_residual_px=float("inf"),
            per_fragment=[],
        )
    pred_centroids_input = np.asarray(
        [predicted_fragment_centroids[fid] for fid in pred_ids])
    # Bring predictions into full-res scan frame so centroid matching is
    # apples-to-apples with gt_centroids_canvas.
    pred_centroids_input_full = pred_centroids_input * s

    # ── Match predicted fragments to GT by full-res scan centroid ────────
    used = set()
    matches_p2g = {}   # pred_id -> gt_idx (or None)
    for pi, pc in zip(pred_ids, pred_centroids_input_full):
        dists = np.linalg.norm(gt_centroids_canvas - pc, axis=1)
        order = np.argsort(dists)
        picked = None
        for gi in order:
            if int(gi) not in used:
                picked = int(gi)
                break
        if picked is not None:
            used.add(picked)
            matches_p2g[pi] = picked
        else:
            matches_p2g[pi] = None

    # ── Project each matched prediction onto the pipeline output canvas ──
    # (still in working-res — Umeyama will absorb the scale difference vs
    # source-frame GT.)
    pred_canvas = []
    gt_source = []
    pairs = []
    for pi, gi in matches_p2g.items():
        if gi is None:
            continue
        T = np.asarray(predicted_transforms[pi], dtype=np.float64)
        if T.shape == (2, 3):
            T3 = np.vstack([T, [0, 0, 1]])
        else:
            T3 = T
        pc_input = np.asarray(predicted_fragment_centroids[pi],
                              dtype=np.float64).reshape(1, 2)
        pc_canvas = _apply_affine_pts(T3, pc_input)[0]
        pred_canvas.append(pc_canvas)
        gt_source.append(gt_centroids_source[gi])
        pairs.append((pi, gi))

    if len(pairs) < 2:
        # Not enough matches even for a similarity fit — return degraded metrics
        return CaseMetrics(
            case_id=truth["case_id"],
            n_fragments_truth=len(gt_frags),
            n_fragments_placed=len(pairs),
            placement_rate=len(pairs) / max(1, len(gt_frags)),
            median_centroid_error_px=float("inf"),
            p90_centroid_error_px=float("inf"),
            median_rotation_error_deg=float("inf"),
            p90_rotation_error_deg=float("inf"),
            mean_iou=0.0,
            alignment_residual_px=float("inf"),
            per_fragment=[],
        )

    pred_canvas = np.asarray(pred_canvas)
    gt_source   = np.asarray(gt_source)

    # Umeyama fit: pipeline-output-canvas → source frame.
    # This absorbs the arbitrary framing of the pipeline's output canvas
    # (translation + rotation + scale).
    M_align = _umeyama_2d(pred_canvas, gt_source)

    # ── Per-fragment errors after alignment ─────────────────────────────
    per_frag = []
    centroid_errs = []
    rotation_errs = []
    ious = []

    src_w, src_h = truth["source_size"]

    for pi, gi in pairs:
        T = np.asarray(predicted_transforms[pi], dtype=np.float64)
        if T.shape == (2, 3):
            T3 = np.vstack([T, [0, 0, 1]])
        else:
            T3 = T
        # Combined: INPUT pixel (working-res scan) → SOURCE frame.
        T_aligned = M_align @ T3

        # Centroid error in source frame.
        pc_input = np.asarray(predicted_fragment_centroids[pi],
                              dtype=np.float64).reshape(1, 2)
        pc_aligned = _apply_affine_pts(T_aligned, pc_input)[0]
        err = float(np.linalg.norm(pc_aligned - gt_centroids_source[gi]))
        centroid_errs.append(err)

        # Rotation error. The pipeline INVERTS the scatter applied by GT,
        # so T_aligned's rotation ≈ -gt_angle. Compare their sum against 0.
        gt_M_full = np.eye(3)
        gt_M_full[:2, :] = np.asarray(
            gt_frags[gi]["affine_source_to_canvas"], dtype=np.float64)
        gt_angle, _   = _decompose_affine(gt_M_full)
        pred_angle, _ = _decompose_affine(T_aligned)
        da = (pred_angle + gt_angle + math.pi) % (2 * math.pi) - math.pi
        rot_err = math.degrees(abs(da))
        rotation_errs.append(rot_err)

        # IoU if we have predicted polygons (rasterised in source frame).
        iou_val = 0.0
        if predicted_fragment_polygons is not None \
                and pi in predicted_fragment_polygons:
            pred_poly_input = np.asarray(
                predicted_fragment_polygons[pi], dtype=np.float64)
            pred_poly_source = _apply_affine_pts(T_aligned, pred_poly_input)

            gt_poly_source = np.asarray(
                gt_frags[gi]["polygon_source"], dtype=np.float64)

            mask_p = _rasterize_polygon_mask(pred_poly_source, src_h, src_w)
            mask_g = _rasterize_polygon_mask(gt_poly_source,   src_h, src_w)
            iou_val = _iou(mask_p, mask_g)
            ious.append(iou_val)

        per_frag.append(PerFragmentMetrics(
            frag_id=int(pi),
            centroid_error_px=err,
            rotation_error_deg=rot_err,
            iou=iou_val,
            matched=True,
        ))

    centroid_errs = np.asarray(centroid_errs)
    rotation_errs = np.asarray(rotation_errs)
    ious_arr = np.asarray(ious) if ious else np.zeros(0)

    return CaseMetrics(
        case_id=truth["case_id"],
        n_fragments_truth=len(gt_frags),
        n_fragments_placed=len(pairs),
        placement_rate=len(pairs) / max(1, len(gt_frags)),
        median_centroid_error_px=float(np.median(centroid_errs)),
        p90_centroid_error_px=float(np.percentile(centroid_errs, 90)),
        median_rotation_error_deg=float(np.median(rotation_errs)),
        p90_rotation_error_deg=float(np.percentile(rotation_errs, 90)),
        mean_iou=float(np.mean(ious_arr)) if len(ious_arr) else 0.0,
        alignment_residual_px=float(np.sqrt(np.mean(
            np.linalg.norm(
                _apply_affine_pts(M_align, pred_canvas) - gt_source,
                axis=1) ** 2))),
        per_fragment=[asdict(p) for p in per_frag],
    )
