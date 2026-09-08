"""Directional morphology, reconstruction and bounded region growing.

The directional closing here is the single operation the whole detector rests on,
so it is worth being precise about what it produces and why the obvious
alternatives do not work.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import config as C

_FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_EIGHT = np.ones((3, 3), dtype=bool)


def _shear_indices(h: int, w: int, slope: float):
    """Map every pixel onto the discrete line it belongs to.

    A line of this family is ``y = y0 + round(x * slope)``, so a pixel's line is
    identified by ``y0 = y - round(x * slope)`` and its position along that line
    is simply ``x``. Shifting each column vertically by ``round(x * slope)``
    therefore straightens every line in the family into a row at once, which is
    what turns a per-line 1-D filter into a single array operation.
    """
    shifts = np.round(np.arange(w) * slope).astype(np.int64)
    lo, hi = int(shifts.min()), int(shifts.max())
    # y - shifts[x] runs over [-hi, h-1-lo], so adding hi puts the first row at 0
    # and makes the sheared array exactly h + hi - lo tall.
    H = h + (hi - lo)
    rows = np.arange(h)[:, None] - shifts[None, :] + hi
    cols = np.broadcast_to(np.arange(w)[None, :], (h, w))
    return rows, cols, H


def _linear_morph_1d(a: np.ndarray, radius: int, closing: bool,
                     slope: float) -> np.ndarray:
    """Closing or opening along one family of discrete lines.

    Invalid cells of the sheared array - the corners that no real pixel maps to -
    are filled with -inf before a max and +inf before a min, and reset in between.
    That reproduces the reference implementation's behaviour exactly: it gathers
    only in-bounds pixels into each line and clamps the window at the ends, and a
    window clamped at the end of a run is the same thing as a window whose
    outside neighbours can never win a max or a min.
    """
    h, w = a.shape
    rows, cols, H = _shear_indices(h, w, slope)
    size = 2 * radius + 1

    valid = np.zeros((H, w), dtype=bool)
    valid[rows, cols] = True

    if closing:
        first, second = ndi.maximum_filter1d, ndi.minimum_filter1d
        fill_a, fill_b = -np.inf, np.inf
    else:
        first, second = ndi.minimum_filter1d, ndi.maximum_filter1d
        fill_a, fill_b = np.inf, -np.inf

    buf = np.full((H, w), fill_a, dtype=np.float64)
    buf[rows, cols] = a

    stage1 = first(buf, size=size, axis=1, mode="constant", cval=fill_a)
    stage1[~valid] = fill_b
    stage2 = second(stage1, size=size, axis=1, mode="constant", cval=fill_b)

    return stage2[rows, cols]


def directional_morph(a: np.ndarray, radius: int, mode: str,
                      orient_count: Optional[int] = None):
    """Linear closing or opening at many orientations; returns (min, max).

    A closing along a line fills any structure narrower than the element. At a
    vessel, the element that happens to lie along it fills nothing while every
    element crossing it fills it completely. At a round lesion, every orientation
    fills it about equally. So:

    ``mn``
        the response every direction agreed on. Large only where the structure is
        bounded in all directions - a lesion. This is the roundness map.
    ``mx``
        the response at least one direction produced. Large wherever anything
        thin exists, vessels included.

    Their difference is the anisotropy, and comparing it to ``mn`` rather than to
    a fixed number is what makes the vessel test scale-free: one rule covers a
    thin peripheral capillary and a wide vein at the disc alike.

    Why not a square element: it removes everything smaller than itself, and a
    microaneurysm is smaller than a vessel is wide, so no size threshold can
    separate the two. That was a real earlier version of this detector and it
    could not be made to work.

    Why not a fixed line kernel
    ---------------------------
    The obvious implementation is one ``strel``-style kernel per angle, applied
    everywhere. It was tried and it is measurably not the same operation. A fixed
    kernel places the element at ``round(t * slope)`` relative to its anchor,
    while gathering a line across the whole image places it at
    ``round((x+t) * slope) - round(x * slope)`` - which depends on where along the
    line the anchor sits. Both are honest 17-pixel approximations to the same
    angle, and they disagree on about 0.6% of pixels, concentrated on exactly the
    high-curvature edges the detector is most sensitive to. That was enough to
    move the seed count on one synthetic retina from 679 pixels to 154 and the
    seed threshold from 8.0 to 6.0.

    So the line-gathering construction is reproduced instead, via the shear in
    :func:`_shear_indices`. It costs a little more and it agrees.
    """
    if orient_count is None:
        orient_count = C.LINEAR_ORIENTATIONS

    src = np.asarray(a, dtype=np.float64)
    radius = max(1, int(radius))
    closing = (mode == "close")

    acc_min = None
    acc_max = None
    for o in range(orient_count):
        theta = np.pi * o / orient_count
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        if abs(cos_t) >= abs(sin_t):
            r = _linear_morph_1d(src, radius, closing, sin_t / cos_t)
        else:
            # Dominant axis is y. Transposing turns it into the case above with
            # slope cos/sin, which is exactly what the reference does.
            r = _linear_morph_1d(src.T, radius, closing, cos_t / sin_t).T

        if acc_min is None:
            acc_min = r.copy()
            acc_max = r.copy()
        else:
            np.minimum(acc_min, r, out=acc_min)
            np.maximum(acc_max, r, out=acc_max)

    if acc_min is None:
        return src.copy(), src.copy()
    return acc_min, acc_max


def square_closing(a: np.ndarray, radius: int) -> np.ndarray:
    """Grey closing by a square element: max filter then min filter.

    A square element is deliberately NOT used for lesion work, for the reason in
    :func:`directional_morph`. It is right here, in the fovea search, because the
    target is a feature many times larger than the vessels being removed.

    Replicate padding, which for a max or a min filter is identical to truncating
    the window at the edge - so the border needs no special case.
    """
    src = np.asarray(a, dtype=np.float32)
    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    d = cv2.dilate(src, k, borderType=cv2.BORDER_REPLICATE)
    e = cv2.erode(d, k, borderType=cv2.BORDER_REPLICATE)
    return e.astype(np.float64)


def hysteresis_mask(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Keep every pixel of ``weak`` that is 8-connected to a pixel of ``strong``.

    Both masks are already gated on the same evidence, so this only ever connects
    what the permissive threshold already believed. What it adds is the
    requirement that a weak pixel be *attached* to a confident one, which is
    exactly what distinguishes the faint continuation of a real vessel from an
    isolated patch of texture at the same amplitude. It is what turns a vessel map
    of disconnected fragments into something shaped like a tree.

    Eight-connectivity, because a vessel crossing the pixel grid at an angle is a
    staircase and four-connectivity breaks it into beads.

    Strong pixels outside the permissive mask are added back, so a seed can never
    be lost to a threshold that was meant to be more forgiving than it.
    """
    strong = np.asarray(strong, dtype=bool)
    weak = np.asarray(weak, dtype=bool)

    lab, _ = ndi.label(weak, structure=_EIGHT)
    keep = np.unique(lab[strong & weak])
    keep = keep[keep > 0]

    out = np.isin(lab, keep) if keep.size else np.zeros_like(weak)
    return out | strong


def flood_bright_region(blur: np.ndarray, inside: np.ndarray,
                        sx: int, sy: int, cut: float,
                        max_r: float, max_area: float) -> Optional[dict]:
    """Grow the bright region containing a seed, under bounds.

    Returns ``None`` when either bound is passed, and that is a real answer, not
    a failure path. Once the fill has run past ``max_r`` or ``max_area`` it has
    left the structure it was asked to measure and is spreading into open retina;
    no number it could return would mean anything. The caller keeps whatever
    estimate it already had, which is worse but honest.

    The radius bound is applied to the candidate set before labelling rather than
    checked afterwards, which is what makes it a genuine barrier: a region cannot
    reach around through distant pixels and come back.
    """
    h, w = blur.shape
    sx, sy = int(round(sx)), int(round(sy))
    if not (0 <= sx < w and 0 <= sy < h):
        return None
    if not inside[sy, sx] or blur[sy, sx] < cut:
        return None

    yy, xx = np.ogrid[:h, :w]
    within = (xx - sx) ** 2 + (yy - sy) ** 2 <= max_r ** 2

    bw = inside & (blur >= cut) & within
    lab, _ = ndi.label(bw, structure=_FOUR)

    seed_label = lab[sy, sx]
    if seed_label == 0:
        return None

    mask = lab == seed_label
    area = int(mask.sum())
    if area == 0 or area > max_area:
        return None                      # escaped: refuse to answer

    ys, xs = np.nonzero(mask)
    return {"mask": mask, "area": area,
            "cx": float(xs.mean()), "cy": float(ys.mean())}
