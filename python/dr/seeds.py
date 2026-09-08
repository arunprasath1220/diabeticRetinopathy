"""Where lesions are, where vessels are, and at what scales.

This module produces the evidence every later decision is made on. Everything
downstream either grows one of these seeds or throws it away; nothing downstream
can invent a lesion the seeder did not find.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .morphology import directional_morph, hysteresis_mask
from .stats import tail_quantile


def round_structure_seeds(green_s: np.ndarray, lum_s: np.ndarray,
                          inside: np.ndarray) -> dict:
    """Directional top-hats: lesion seeds, the vessel map, and the scale ladder.

    Vessels vanish because they survive a closing along their own direction, the
    macula vanishes because its gradual ramp barely responds, and a lesion sitting
    on a vessel is still found because the lesion itself is round regardless of
    what it touches. That last property is the whole reason for the directional
    element: a square one removes every small lesion along with the vessels,
    because a microaneurysm is smaller than a vessel is wide.

    The vessel map is derived from the same evidence rather than from a separate
    pass, which is both cheaper and sounder. The previous version thresholded each
    response against an absolute level and recovered under a quarter of the vessel
    tree in 755 disconnected fragments; three quarters of the vasculature was
    simply not being suppressed, which is where the marks on healthy retina were
    coming from.

    Why seeding stays on the shortest element
    -----------------------------------------
    The obvious improvement was measured and rejected. A closing only fills a
    structure narrower than the element, so a blot hemorrhage wider than the
    element barely responds: on a planted hemorrhage of radius 20 the mean top-hat
    at the nominal element is 0.4 grey levels and not one of its 1257 pixels
    seeds, while at twice and four times that length it answers at 25.4 and 34.7.

    Seeding from the longer rungs as well does fix the seeding, and does not fix
    the detection - the newly seeded regions are rejected at the contrast test
    instead, because a long element also responds to the macula and to
    illumination falloff, which arrive together with the hemorrhages. Over three
    retinas it raised false marks on healthy images from 19 to 29 while recovering
    one large hemorrhage out of six. The gap is real, but it lives further down in
    how extent and contrast are measured, and adding scales to the seeder only
    moves the failure rather than removing it.
    """
    h, w = green_s.shape
    se_l = max(3, int(round(min(w, h) * C.LINEAR_SE_FRAC)))

    closed_min, closed_max = directional_morph(green_s, se_l, "close",
                                               C.LINEAR_ORIENTATIONS)
    _, opened_max = directional_morph(lum_s, se_l, "open",
                                      C.LINEAR_ORIENTATIONS)

    # Filled by every direction: round. This is the lesion evidence.
    round_dark = np.maximum(0.0, closed_min - green_s)
    round_bright = np.maximum(0.0, lum_s - opened_max)
    # Spread across orientations: near zero on a round blob, large on a vessel.
    aniso_dark = np.maximum(0.0, closed_max - closed_min)

    round_dark[~inside] = 0
    round_bright[~inside] = 0
    aniso_dark[~inside] = 0

    # ---- The saturation ladder ------------------------------------------
    # The same two top-hats at successively longer elements. Everything bounded
    # has already been filled at its own scale and does not respond any harder at
    # the next rung; everything that continues past the element does. The
    # difference between consecutive rungs is the saturation test.
    sat_se = [se_l]
    for k in range(1, C.SATURATION_LADDER):
        sat_se.append(max(sat_se[-1] + 1,
                          int(round(se_l * C.SATURATION_SE_MULT ** k))))

    sat_dark = [round_dark]
    sat_bright = [round_bright]
    for k in range(1, len(sat_se)):
        c_min, _ = directional_morph(green_s, sat_se[k], "close",
                                     C.SATURATION_ORIENTATIONS)
        _, o_max = directional_morph(lum_s, sat_se[k], "open",
                                     C.SATURATION_ORIENTATIONS)
        dk = np.maximum(0.0, c_min - green_s)
        br = np.maximum(0.0, lum_s - o_max)
        dk[~inside] = 0
        br[~inside] = 0
        sat_dark.append(dk)
        sat_bright.append(br)

    # ---- Seeds ------------------------------------------------------------
    t_dark = max(C.SEED_FLOOR_DARK,
                 tail_quantile(round_dark, inside, C.SEED_TAIL_Q))
    t_bright = max(C.SEED_FLOOR_BRIGHT,
                   tail_quantile(round_bright, inside, C.SEED_TAIL_Q))

    dark_seed = inside & (round_dark > t_dark)
    bright_seed = inside & (round_bright > t_bright)

    # ---- Vessel map -------------------------------------------------------
    # Found at a strict level and followed outward at a permissive one, so a
    # vessel is traced along its length instead of appearing only where it happens
    # to be darkest. The elongation test comes first: a lesion fails it however
    # dark it is, which is what stops the vessel map from swallowing the findings
    # it exists to protect.
    t_vessel_seed = max(3.0, tail_quantile(aniso_dark, inside, C.VESSEL_SEED_Q))
    t_vessel_grow = max(2.0, tail_quantile(aniso_dark, inside, C.VESSEL_GROW_Q))

    elongated = inside & (aniso_dark > C.VESSEL_ELONGATION * round_dark)
    v_seed = elongated & (aniso_dark > t_vessel_seed)
    v_grow = elongated & (aniso_dark > t_vessel_grow)
    vessel_mask = hysteresis_mask(v_seed, v_grow)

    return {
        "darkSeed": dark_seed, "brightSeed": bright_seed,
        "vesselMask": vessel_mask, "seL": se_l,
        "satSE": sat_se, "satDark": sat_dark, "satBright": sat_bright,
        "roundDark": round_dark, "roundBright": round_bright,
        "tDark": t_dark, "tBright": t_bright,
    }
