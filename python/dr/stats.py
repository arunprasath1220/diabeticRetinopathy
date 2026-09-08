"""Robust statistics, computed exactly the way the JavaScript computes them.

Both routines here could be one call to ``np.median`` or ``np.percentile``, and
both are written out as histograms instead. That is deliberate.

The JavaScript bins because it has to stay a single linear pass in a browser.
Binning changes the answer slightly - the median lands on a bin centre rather
than on a data value - and every detection threshold in the pipeline is derived
from these two numbers. Using the exact statistic here would shift those
thresholds by up to half a bin, which is enough to move borderline candidates in
or out and make the two implementations disagree about a lesion count.

So the bin edges, the clamping and the subsampling stride are all reproduced. The
cost is a few lines; the gain is that a disagreement between the Python and the
JavaScript is a real bug rather than a rounding difference nobody can chase.
"""

from __future__ import annotations

import numpy as np


def _sampled(values: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    """The pixels the JavaScript actually visits.

    It walks ``for (i = 0; i < n; i += step)`` in row-major order and skips
    anything outside the mask. Since these arrays are C-ordered, ``ravel()[::step]``
    visits exactly the same pixels in exactly the same order.

    A noise scale does not need every pixel; a few hundred thousand samples fix
    it as precisely as the full image does.
    """
    step = 2 if n > 300_000 else 1
    v = values.ravel()[::step]
    m = mask.ravel()[::step]
    return v[m]


def robust_sigma(dev: np.ndarray, mask: np.ndarray) -> float:
    """Noise scale of a deviation map: 1.4826 * MAD, floored at half a level.

    The standard deviation is the wrong statistic here, and using it was a real
    bug rather than a stylistic choice. The vessel tree is a large population of
    strong dark deviations; it inflates the standard deviation, which pushes
    every threshold derived from it upward, which hides exactly the faint lesions
    the detector exists to find. The median absolute deviation ignores that
    minority and tracks the actual noise floor.

    The 1.4826 makes MAD a consistent estimator of sigma for Gaussian data, so
    the multipliers in :mod:`dr.config` can be read as "so many sigma" and mean it.
    """
    dev = np.asarray(dev, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n = dev.size

    v = _sampled(dev, mask, n)
    count = v.size
    if count == 0:
        return 1.0

    BINS, LO, HI = 320, -80.0, 80.0
    sc = BINS / (HI - LO)

    clamped = np.clip(v, LO, 79.999)
    idx = ((clamped - LO) * sc).astype(np.int64)
    np.clip(idx, 0, BINS - 1, out=idx)
    hist = np.bincount(idx, minlength=BINS)

    half = count / 2.0
    b = int(np.searchsorted(np.cumsum(hist), half, side="left"))
    b = min(b, BINS - 1)
    median = LO + (b + 0.5) / sc

    ABINS, AHI = 320, 80.0
    asc = ABINS / AHI
    a = np.abs(v - median)
    np.clip(a, 0, 79.999, out=a)
    aidx = (a * asc).astype(np.int64)
    np.clip(aidx, 0, ABINS - 1, out=aidx)
    ahist = np.bincount(aidx, minlength=ABINS)

    acum = np.cumsum(ahist)
    if acum[-1] < half:
        mad = AHI
    else:
        ab = int(np.searchsorted(acum, half, side="left"))
        ab = min(ab, ABINS - 1)
        mad = (ab + 0.5) / asc

    return max(0.5, 1.4826 * mad)


def tail_quantile(values: np.ndarray, mask: np.ndarray, q: float) -> float:
    """Upper-quantile anchor for a one-sided response map.

    :func:`robust_sigma` is the right tool for the symmetric deviation maps, where
    the median really is the background and the MAD really is the noise. It is the
    wrong tool for a top-hat response, which is non-negative and piles most of its
    mass on exactly zero: there the MAD describes that zero atom and says nothing
    whatever about the tail the threshold has to sit in. This reads the tail
    directly.
    """
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n = values.size

    v = _sampled(values, mask, n)
    count = v.size
    if count == 0:
        return 1.0

    BINS, HI = 2048, 128.0
    sc = BINS / HI

    clamped = np.clip(v, 0.0, HI - 0.001)
    idx = (clamped * sc).astype(np.int64)
    np.clip(idx, 0, BINS - 1, out=idx)
    hist = np.bincount(idx, minlength=BINS)

    target = count * q
    cum = np.cumsum(hist)
    if cum[-1] < target:
        return HI
    b = int(np.searchsorted(cum, target, side="left"))
    b = min(b, BINS - 1)
    return (b + 0.5) / sc


def cam_field_stats(cam: np.ndarray, mask: np.ndarray) -> dict:
    """Median and robust spread of a normalised activation map, over retina only.

    These are what tell a focal hotspot from a map that is simply bright
    everywhere: the peak is 1.0 either way, but only in the first case is the
    typical retinal pixel far below it.
    """
    cam = np.asarray(cam, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n = cam.size

    v = _sampled(cam, mask, n)
    count = v.size
    if count == 0:
        return {"median": 0.0, "mad": 0.0}

    BINS = 512
    idx = (v * BINS).astype(np.int64)
    np.clip(idx, 0, BINS - 1, out=idx)
    hist = np.bincount(idx, minlength=BINS)

    half = count / 2.0
    b = int(np.searchsorted(np.cumsum(hist), half, side="left"))
    b = min(b, BINS - 1)
    median = (b + 0.5) / BINS

    a = np.abs(v - median)
    aidx = (a * BINS).astype(np.int64)
    np.clip(aidx, 0, BINS - 1, out=aidx)
    ahist = np.bincount(aidx, minlength=BINS)

    acum = np.cumsum(ahist)
    if acum[-1] < half:
        mad = 1.0
    else:
        ab = int(np.searchsorted(acum, half, side="left"))
        ab = min(ab, BINS - 1)
        mad = (ab + 0.5) / BINS

    return {"median": median, "mad": 1.4826 * mad}
