"""The whole screening pipeline, Step 1 through Step 5.

Structure, and why it is this way
---------------------------------
Steps 1, 3, 4 and 5 are classical image processing. Step 2 (classification) and
Step 3a (Grad-CAM) need a network. They are deliberately decoupled: a missing or
failed model costs the classifier and the heatmap and nothing else, so a district
deployment with no model file still produces a lesion map and an ICDR estimate.

What it does not produce is the cross-check between the two, and
:func:`dr.grading.assess_severity` records that absence as a reason for caution
rather than letting it pass silently.

The quality gate is a hard stop, not a warning. An image that cannot be assessed
produces no grade at all, because a confident-looking result with nothing
underneath it is the one that gets acted on in the field.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from . import config as C
from .anatomy import estimate_anatomy
from .enhance import enhance_image
from .grading import assess_severity
from .lesions import detect_lesion_candidates
from .quality import assess_quality, compute_metrics


def to_working_size(rgb: np.ndarray) -> np.ndarray:
    """Cap the long edge.

    Every threshold in the pipeline that is expressed in pixels was measured at
    this scale, so this is not merely a performance choice - running at native
    resolution would silently change what counts as a microaneurysm.
    """
    a = np.asarray(rgb)
    h, w = a.shape[:2]
    scale = min(1.0, C.WORKING_MAX_DIM / max(h, w))
    if scale >= 1.0:
        return a
    return cv2.resize(a, (max(1, int(round(w * scale))),
                          max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)


def run_pipeline(rgb: np.ndarray, classifier=None,
                 verbose: bool = False) -> dict:
    """Run every stage and return everything, including what did not run."""
    t_start = time.perf_counter()
    result: dict = {"timings": {}}

    def say(msg: str) -> None:
        if verbose:
            print(msg)

    working = to_working_size(rgb)
    result["working"] = working

    # ---- STEP 1: quality --------------------------------------------------
    t0 = time.perf_counter()
    metrics = compute_metrics(working)
    quality = assess_quality(metrics)
    result["metrics"] = metrics
    result["quality"] = quality
    result["timings"]["quality"] = (time.perf_counter() - t0) * 1000
    say(f"Step 1  quality: {quality['verdict']}")

    if quality["verdict"] == "reject":
        result["stopped"] = True
        result["processed"] = working
        result["enhanced"] = False
        result["anatomy"] = None
        result["grading"] = None
        result["lesions"] = None
        sev = assess_severity(None, None, None, quality)
        sev["reason"] = ("This image could not be assessed at all, so nothing "
                         "here rules anything out. " + quality["reason"])
        result["severity"] = sev
        result["timings"]["total"] = (time.perf_counter() - t_start) * 1000
        say(f"  rejected: {quality['reason']}")
        return result

    result["stopped"] = False

    # ---- STEP 1b: enhancement ---------------------------------------------
    t0 = time.perf_counter()
    if quality["verdict"] == "enhance":
        processed = enhance_image(working)
        result["enhanced"] = True
    else:
        processed = working
        result["enhanced"] = False
    result["processed"] = processed
    result["timings"]["enhance"] = (time.perf_counter() - t0) * 1000

    # ---- STEP 3 (first): anatomical landmarks -----------------------------
    # Computed before Grad-CAM because both the quadrant naming in Step 3 and the
    # disc exclusion in Step 4 depend on them.
    t0 = time.perf_counter()
    anatomy: Optional[dict] = None
    try:
        anatomy = estimate_anatomy(processed)
        grown = "grown" if anatomy["disc"]["measuredExtent"] else "plateau"
        say(f"Step 3  disc r={anatomy['disc']['radius']:.1f} px ({grown}), "
            f"fovea at {anatomy['fovea']['x']:.0f},{anatomy['fovea']['y']:.0f}")
    except Exception as err:                              # noqa: BLE001
        result["anatomyError"] = str(err)
        say(f"Step 3  landmark estimation failed: {err}")
    result["anatomy"] = anatomy
    result["timings"]["anatomy"] = (time.perf_counter() - t0) * 1000

    # ---- STEP 2: classification -------------------------------------------
    result["modelAvailable"] = classifier is not None
    if classifier is not None:
        t0 = time.perf_counter()
        try:
            result["grading"] = classifier.classify(processed)
            say(f"Step 2  P(DR) = {result['grading']['probs'][1]:.3f}")
        except Exception as err:                          # noqa: BLE001
            result["grading"] = None
            result["gradingError"] = str(err)
            say(f"Step 2  classification failed: {err}")
        result["timings"]["classify"] = (time.perf_counter() - t0) * 1000
    else:
        result["grading"] = None
        say("Step 2  skipped — no classifier available")

    # ---- STEP 3a: Grad-CAM -------------------------------------------------
    # Always for the "DR present" class, whatever was predicted: what a clinician
    # needs is not why the model said "no disease", but where the evidence for
    # disease would have been.
    if classifier is not None and result.get("grading") is not None:
        t0 = time.perf_counter()
        try:
            cam = classifier.grad_cam(processed, 1, anatomy)
            result["cam"] = cam
            say(f"Step 3a Grad-CAM peak in {cam['description']['region']}, "
                f"concentration {cam['description']['concentration']:.2f}")
        except Exception as err:                          # noqa: BLE001
            result["camError"] = str(err)
            say(f"Step 3a Grad-CAM failed: {err}")
        result["timings"]["gradcam"] = (time.perf_counter() - t0) * 1000

    # ---- STEP 4: lesion candidates ----------------------------------------
    t0 = time.perf_counter()
    try:
        lesions = detect_lesion_candidates(processed, anatomy)
        result["lesions"] = lesions
        c = lesions["counts"]
        say(f"Step 4  {len(lesions['candidates'])} candidates "
            f"(MA {c['MA']}, HEM {c['HEM']}, HE {c['HE']}, CWS {c['CWS']})")
    except Exception as err:                              # noqa: BLE001
        result["lesions"] = None
        result["lesionError"] = str(err)
        say(f"Step 4  lesion detection failed: {err}")
    result["timings"]["lesions"] = (time.perf_counter() - t0) * 1000

    # ---- STEP 5: ICDR severity ---------------------------------------------
    result["severity"] = assess_severity(result["lesions"], anatomy,
                                         result.get("grading"), quality)
    sev = result["severity"]
    say(f"Step 5  level {sev['level']} — {sev['label']} (refer: {sev['refer']})")

    result["timings"]["total"] = (time.perf_counter() - t_start) * 1000
    return result
