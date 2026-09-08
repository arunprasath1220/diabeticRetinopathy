"""Step 1: measure whether this image can be assessed at all, and decide."""

from __future__ import annotations

import cv2
import numpy as np

from . import config as C


def compute_metrics(rgb: np.ndarray) -> dict:
    """Focus, exposure and field-of-view coverage.

    Everything is measured on a copy resized to a fixed edge, so the thresholds in
    :func:`assess_quality` mean the same thing regardless of the camera's native
    resolution. A focus score computed at native resolution is not comparable
    between a 6 MP handheld and a 1 MP phone adapter, and would gate the two
    differently for no clinical reason.
    """
    a = np.asarray(rgb)
    h, w = a.shape[:2]
    scale = min(1.0, C.METRIC_DIM / max(h, w))
    if scale < 1.0:
        small = cv2.resize(a, (max(1, int(round(w * scale))),
                               max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = a

    g = small.astype(np.float64)
    gray = 0.299 * g[:, :, 0] + 0.587 * g[:, :, 1] + 0.114 * g[:, :, 2]

    # Focus: variance of the Laplacian. A sharp image has strong second
    # derivatives; a blurred one does not. The variance rather than the mean,
    # because the mean of a Laplacian is near zero on any image and says nothing.
    lap = (gray[1:-1, :-2] + gray[1:-1, 2:] +
           gray[:-2, 1:-1] + gray[2:, 1:-1] - 4 * gray[1:-1, 1:-1])
    focus_var = float(lap.var()) if lap.size else 0.0

    return {
        "focusVar": focus_var,
        "brightMean": float(gray.mean()),
        "brightStd": float(gray.std()),
        # Field of view: how much of the frame is not black border.
        "fov": float((gray > 15).sum() / gray.size),
    }


def assess_quality(m: dict) -> dict:
    """The gradable / ungradable decision.

    Three verdicts: ``reject`` stops the pipeline, ``enhance`` runs the
    enhancement stage first, ``ok`` analyses as captured.

    The rejection path matters more than it looks. Screening an unassessable
    image produces a confident-looking result with nothing underneath it, and in a
    rural screening programme that result is the one that gets acted on. The
    reasons are phrased as instructions to whoever is holding the camera, because
    that is the only person who can fix the problem.
    """
    T = C.QUALITY_THRESHOLDS

    if m["fov"] < T["minFov"]:
        return _reject(
            f"Insufficient retinal field of view (only {m['fov']*100:.0f}% of the "
            "frame is non-black) — recentre the eye in the camera and recapture.")
    if m["brightMean"] < T["minBrightMean"]:
        return _reject(
            f"Mean brightness too low ({m['brightMean']:.1f}/255) — recapture "
            "with more illumination.")
    if m["brightMean"] > T["maxBrightMean"]:
        return _reject(
            f"Image overexposed (mean brightness {m['brightMean']:.1f}/255) — "
            "reduce flash intensity and recapture.")
    if m["focusVar"] < T["minFocusVar"]:
        return _reject(
            f"Image too blurry (focus score {m['focusVar']:.1f}) — hold the "
            "camera steady and refocus before recapture.")

    reasons = []
    if m["focusVar"] < T["enhanceFocusVar"]:
        reasons.append("borderline focus")
    if m["brightMean"] < T["enhanceBrightLow"] or m["brightMean"] > T["enhanceBrightHigh"]:
        reasons.append("borderline exposure")
    if m["fov"] < T["enhanceFov"]:
        reasons.append("limited field of view")

    if reasons:
        return {"verdict": "enhance", "reason": ", ".join(reasons),
                "reasons": reasons}
    return {"verdict": "ok", "reason": "", "reasons": []}


def _reject(reason: str) -> dict:
    return {"verdict": "reject", "reason": reason, "reasons": []}
