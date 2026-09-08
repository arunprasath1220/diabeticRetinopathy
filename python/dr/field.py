"""Camera aperture geometry and the usable retinal field."""

from __future__ import annotations

import numpy as np

from .imaging import erode_mask


def aperture_geometry(lum: np.ndarray) -> dict:
    """Centre and radius of the camera aperture, from the extent of the lit area.

    The cutoff is deliberately firm - a fifth of the frame's peak - so that the
    dim vignetted rim does not stretch the fitted circle outward. Deciding which
    pixels are *inside* that aperture is a different question with a different
    threshold; see :func:`retina_field_mask`.
    """
    lum = np.asarray(lum, dtype=np.float64)
    h, w = lum.shape
    hi = float(lum.max())
    thr = max(14.0, hi * 0.20)

    lit = lum > thr
    if not lit.any():
        return {"cx": w / 2, "cy": h / 2, "R": min(w, h) / 2}

    cols = np.nonzero(lit.any(axis=0))[0]
    rows = np.nonzero(lit.any(axis=1))[0]
    min_x, max_x = int(cols[0]), int(cols[-1])
    min_y, max_y = int(rows[0]), int(rows[-1])

    return {"cx": (min_x + max_x) / 2,
            "cy": (min_y + max_y) / 2,
            "R": max(1.0, max(max_x - min_x, max_y - min_y) / 2)}


def retina_field_mask(lum: np.ndarray, erode_frac: float) -> np.ndarray:
    """The usable retina, found by fitting the aperture rather than trimming.

    Two shortcuts were tried before this and both failed on real images. A fixed
    brightness cutoff keeps the vignetted rim, which is genuine retina but far
    darker than the rest, so it reads as one enormous dark lesion - the crescent
    that appeared beside the disc. A fixed erosion cannot reach that rim either,
    because the vignette is much wider than any sensible margin. Fitting the
    aperture and working inside a fraction of its radius handles both, and adapts
    to how much of the frame the retina fills.

    Two thresholds, for two different jobs. Fitting the circle wants a firm
    cutoff so the dim rim does not stretch it. Deciding membership wants a far
    lower one, because a hemorrhage is dark: judged at the fitting cutoff it falls
    below threshold and is carved out of the field as though it were outside the
    camera's view, so every dark lesion punched a hole in its own analysis region
    and could never be found.

    The fraction kept is deliberately generous. Trimming a tenth of the radius was
    measured to discard genuine peripheral lesions, so large rim artifacts are
    dealt with by a size-keyed rule during classification instead.
    """
    lum = np.asarray(lum, dtype=np.float64)
    h, w = lum.shape
    hi = float(lum.max())

    fit_thr = max(14.0, hi * 0.20)
    member_thr = max(6.0, hi * 0.06)

    lit = lum > fit_thr
    if not lit.any():
        return np.zeros((h, w), dtype=bool)

    cols = np.nonzero(lit.any(axis=0))[0]
    rows = np.nonzero(lit.any(axis=1))[0]
    cx = (int(cols[0]) + int(cols[-1])) / 2
    cy = (int(rows[0]) + int(rows[-1])) / 2
    R = max(int(cols[-1]) - int(cols[0]), int(rows[-1]) - int(rows[0])) / 2

    r_in = R * (1 - max(0.02, erode_frac * 0.9))

    yy, xx = np.ogrid[:h, :w]
    in_circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_in ** 2

    mask = in_circle & (lum > member_thr)     # true black surround only
    return erode_mask(mask, 2)
