"""The peripapillary bright-suppression rules.

These mirror the JavaScript harness one for one, so a regression in either
implementation shows up as a disagreement between them rather than as a quiet
drift in one.

The test that matters most is :func:`test_colour_separates_disc_from_exudate`.
Two marks at the same distance from the disc, the same size, the same brightness
above background, differing only in colour: one must be rejected and one kept. A
rule that removes both is not an improvement, it is a bigger exclusion zone, and
only a paired test can tell those apart.
"""

from __future__ import annotations

import numpy as np
import pytest

import dr
from dr.anatomy import estimate_anatomy
from dr.lesions import detect_lesion_candidates

from .synthetic import DISC, synthetic_fundus


def analyse(img):
    anatomy = estimate_anatomy(img)
    return detect_lesion_candidates(img, anatomy), anatomy


def found(lesions, mark):
    return any(np.hypot(c["cx"] - mark["x"], c["cy"] - mark["y"]) <= mark["r"] + 8
               for c in lesions["candidates"])


def bright_near_disc(lesions, mult):
    return [c for c in lesions["candidates"]
            if c["type"] in ("HE", "CWS")
            and np.hypot(c["cx"] - DISC["x"], c["cy"] - DISC["y"]) <= DISC["r"] * mult]


def test_disc_extent_is_grown_not_inferred():
    img = synthetic_fundus(disc=DISC)
    disc = dr.estimate_optic_disc(img)

    assert disc["measuredExtent"], "extent should be grown on a clean disc"
    assert disc["tissue"] is not None, "a halo mask should have been produced"
    # Within 20% of truth. The grown radius sits slightly wide because the halo
    # level deliberately reaches past the rim.
    assert abs(disc["radius"] - DISC["r"]) / DISC["r"] < 0.20


def test_crescent_is_not_reported_as_exudate():
    """A scleral crescent hugging the temporal rim.

    Bright, round, sharply bounded, and entirely not a lesion.
    """
    img = synthetic_fundus(
        disc=DISC,
        crescent={"x": DISC["x"], "y": DISC["y"], "r": DISC["r"] + 12,
                  "from": 5.1, "to": 6.28})
    lesions, _ = analyse(img)

    near = bright_near_disc(lesions, 3.0)
    assert not near, f"{len(near)} false exudate(s) reported at the disc margin"


def test_colour_separates_disc_from_exudate():
    """The paired test: same distance, same size, same brightness."""
    D = DISC["r"] * 1.9
    white = {"x": DISC["x"], "y": DISC["y"] - D, "r": 9,
             "colour": (232.0, 228.0, 224.0)}    # reflex / atrophy
    yellow = {"x": DISC["x"], "y": DISC["y"] + D, "r": 9,
              "colour": (244.0, 226.0, 116.0)}   # lipid exudate

    img = synthetic_fundus(disc=DISC, spots=[white, yellow])
    lesions, _ = analyse(img)

    assert not found(lesions, white), \
        "the white peripapillary mark was reported as an exudate"
    assert found(lesions, yellow), \
        ("the yellow peripapillary exudate was lost — the rule is suppressing on "
         "position rather than on colour")


def test_circinate_ring_survives():
    """The case the fix is most likely to damage.

    Real exudate at 1.6-2.4 disc radii, well beyond the rim. It must survive.
    """
    ring = []
    for k in range(6):
        a = k * np.pi / 3 + 0.3
        d = DISC["r"] * (1.6 + 0.8 * ((k % 3) / 2))
        ring.append({"x": DISC["x"] + d * np.cos(a),
                     "y": DISC["y"] + d * np.sin(a), "r": 8})

    img = synthetic_fundus(disc=DISC, exudates=ring)
    lesions, _ = analyse(img)

    n_found = sum(1 for e in ring if found(lesions, e))
    assert n_found >= 5, f"only {n_found} of {len(ring)} circinate exudates survived"


def test_dark_mark_beside_disc_is_exempt():
    """Nothing about the disc is dark.

    The peripapillary rules must leave a dark mark alone. This asserts the rules
    did not fire, which is separable from whether the detector found the mark.
    """
    spot = {"x": DISC["x"] + DISC["r"] * 1.9, "y": DISC["y"], "r": 9,
            "colour": (64.0, 22.0, 20.0)}
    img = synthetic_fundus(disc=DISC, spots=[spot])
    lesions, _ = analyse(img)

    assert lesions["rejected"]["discTissue"] == 0, \
        "a dark mark was rejected as disc tissue"
    assert lesions["rejected"]["discColour"] == 0, \
        "a dark mark was rejected on disc colour"


def test_clean_retina_invents_nothing():
    img = synthetic_fundus(disc=DISC)
    lesions, _ = analyse(img)
    n = len(lesions["candidates"])
    assert n <= 2, f"{n} candidates on a clean synthetic retina"


def test_pipeline_runs_end_to_end_without_a_model():
    """A missing classifier costs Steps 2 and 3a and nothing else."""
    img = synthetic_fundus(disc=DISC)
    R = dr.run_pipeline(img)

    assert not R["stopped"]
    assert R["lesions"] is not None
    assert R["severity"]["level"] is not None
    assert R["grading"] is None
    assert any("classifier did not run" in d for d in R["severity"]["doubts"]), \
        "a missing classifier must be recorded as a reason for caution"
