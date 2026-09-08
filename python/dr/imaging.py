"""Shared image primitives.

Arrays are (H, W) float64 in row-major order, which means ``a.ravel()[y*w + x]``
is the same pixel the JavaScript reaches with ``arr[y*w + x]``. That alignment is
deliberate: it lets every routine here be checked against the reference
implementation index by index rather than only in aggregate.

Pixel coordinates are 0-based throughout, as in the JavaScript. (The MATLAB port
is 1-based, because MATLAB is; that is the one place where the three
implementations report different numbers for the same pixel.)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def image_channels(rgb: np.ndarray) -> dict:
    """Pull the channels the pipeline reasons about out of an RGB image.

    Returns luminance and green - green carries the strongest blood contrast and
    is what every dark-lesion test is measured on - plus red and blue.

    Nothing in the morphology uses red or blue. They are carried because telling
    lipid exudate from optic disc tissue is a question about hue, not about
    shape, and needs all three.
    """
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("expected an H x W x 3 RGB image")

    r = a[:, :, 0].astype(np.float64)
    g = a[:, :, 1].astype(np.float64)
    b = a[:, :, 2].astype(np.float64)
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    h, w = lum.shape
    return {"h": h, "w": w, "n": h * w,
            "lum": lum, "green": g, "red": r, "blue": b}


def box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2*radius+1) square window, replicating at the border.

    The padding is not a detail. Zero padding near the edge of a fundus aperture
    mixes the black surround into every window, drags the local background down,
    and makes ordinary edge retina read as abnormally bright - a ring of spurious
    findings around the border. Replicate padding removes that for a rectangular
    border; :func:`masked_blur` removes it for the circular one, which is the
    boundary that actually matters here.
    """
    radius = int(radius)
    if radius <= 0:
        return np.asarray(a, dtype=np.float64).copy()
    return ndi.uniform_filter(np.asarray(a, dtype=np.float64),
                              size=2 * radius + 1, mode="nearest")


def masked_blur(a: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Local mean over the masked pixels only.

    Averages over a square window but counts only pixels where ``mask`` is true,
    renormalising each window by how many of those it actually contained. Where a
    window holds no masked pixel at all the original value is passed through.

    This is the background estimator the whole detector rests on. Dividing the
    blurred signal by the blurred mask renormalises each window to the pixels
    that carry image data, which removes the bright ring at the aperture without
    eroding the field and losing genuine peripheral lesions.
    """
    a = np.asarray(a, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)

    num = box_blur(a * m, radius)
    den = box_blur(m, radius)

    out = a.copy()
    valid = den > 1e-3
    out[valid] = num[valid] / den[valid]
    return out


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode a binary mask by a (2*radius+1) square.

    Implemented as a threshold on the box-blurred mask, which is exactly what the
    JavaScript does: a box-blurred binary image equals 1 only where every pixel
    in the window is 1, so this is an exact erosion by a square element - and it
    inherits the replicate padding, which matters because the aperture can run
    off the edge of the frame.
    """
    radius = int(radius)
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    b = box_blur(np.asarray(mask, dtype=np.float64), radius)
    return b >= 0.999


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a binary mask by a (2*radius+1) square.

    Used to turn the vessel map into a vessel *zone*: a lesion sitting against a
    vessel must still be allowed to touch it, so the zone is only a pixel or two
    wider than the map itself.
    """
    radius = int(radius)
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    win = (2 * radius + 1) ** 2
    b = box_blur(np.asarray(mask, dtype=np.float64), radius)
    return b >= 0.5 / win


def normalize_inside(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scale to 0..1 using only the masked pixels' range, zero elsewhere.

    Normalising over the whole frame instead would let the black surround set the
    bottom of the range, compressing everything the retina actually contains into
    the top of the scale and flattening the very differences being measured.
    """
    a = np.asarray(a, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(a)

    v = a[mask]
    lo = float(v.min())
    hi = float(v.max())
    rng = (hi - lo) or 1.0

    out = np.zeros_like(a)
    out[mask] = (a[mask] - lo) / rng
    return out


def gradient_magnitude(a: np.ndarray) -> np.ndarray:
    """Central-difference edge strength, zero on the one-pixel border.

    Hard exudates have sharp margins and cotton wool spots do not, which is the
    only thing that separates them once size and colour have been accounted for.
    """
    a = np.asarray(a, dtype=np.float64)
    h, w = a.shape
    grad = np.zeros((h, w), dtype=np.float64)
    if h > 2 and w > 2:
        gx = a[1:-1, 2:] - a[1:-1, :-2]
        gy = a[2:, 1:-1] - a[:-2, 1:-1]
        grad[1:-1, 1:-1] = np.hypot(gx, gy)
    return grad
