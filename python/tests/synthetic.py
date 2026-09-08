"""Build fundus images with known ground truth.

This exists because the repository holds no real fundus images, and a detector
tuned only against images you cannot redistribute is a detector nobody else can
check. A synthetic scene is weaker evidence than a real retina - the noise is
wrong, the texture is wrong, the vessels are too regular - but it does establish
that a specific rule fires on a specific structure, which is what the tests here
assert and all they claim.

The colours matter and are not arbitrary. Retinal background is red-orange; the
disc is pale and desaturated; exudate is yellow; sclera and reflex are white-grey.
The peripapillary colour test in :mod:`dr.lesions` turns on exactly that
difference, so a test image that got it wrong would pass a broken detector.
"""

from __future__ import annotations

import numpy as np

DISC = {"x": 200.0, "y": 300.0, "r": 50.0}

RETINA = (178.0, 82.0, 44.0)      # red-orange background
DISC_COLOUR = (236.0, 226.0, 216.0)   # pale, desaturated
CRESCENT = (214.0, 202.0, 196.0)      # scleral / atrophic white-grey
EXUDATE = (246.0, 228.0, 108.0)       # lipid yellow
VESSEL = (96.0, 26.0, 22.0)


def synthetic_fundus(size=(600, 600), disc=None, crescent=None,
                     spots=None, exudates=None, vessels=True):
    """Return an H x W x 3 uint8 fundus image."""
    H, W = size
    rng = np.random.default_rng(7)

    yy, xx = np.mgrid[:H, :W]
    cx, cy = W / 2, H / 2
    R = min(W, H) * 0.47
    d = np.hypot(xx - cx, yy - cy)

    vig = 1 - 0.30 * (d / R) ** 3
    noise = (rng.random((H, W)) - 0.5) * 6
    img = np.stack([RETINA[k] * vig + noise for k in range(3)], axis=2)
    img[d > R] = 0

    if disc is not None:
        img = _blob(img, xx, yy, disc["x"], disc["y"], disc["r"], 8, DISC_COLOUR)

    if crescent is not None:
        img = _arc(img, xx, yy, crescent["x"], crescent["y"], crescent["r"], 5,
                   CRESCENT, crescent["from"], crescent["to"])

    for s in (spots or []):
        img = _blob(img, xx, yy, s["x"], s["y"], s["r"], 3, s["colour"])

    for e in (exudates or []):
        img = _blob(img, xx, yy, e["x"], e["y"], e["r"], 2, EXUDATE)

    if vessels and disc is not None:
        for t in range(420):
            for sgn in (1, -1):
                ang = sgn * 0.55
                px = int(round(disc["x"] + t * np.cos(ang)))
                py = int(round(disc["y"] + t * np.sin(ang)))
                for o in (-2, -1, 0, 1, 2):
                    y = py + o
                    if px < 0 or px >= W or y < 0 or y >= H:
                        continue
                    a = 0.9 if o == 0 else 0.5
                    img[y, px] = img[y, px] * (1 - a) + np.array(VESSEL) * a

    return np.clip(img, 0, 255).astype(np.uint8)


def _blob(img, xx, yy, bx, by, r, soft, colour):
    """A soft-edged disc: full strength inside r, falling off over ``soft``."""
    d = np.hypot(xx - bx, yy - by)
    a = np.zeros(d.shape)
    a[d <= r] = 1.0
    ring = (d > r) & (d <= r + soft)
    a[ring] = 1 - (d[ring] - r) / soft
    a3 = a[:, :, None]
    return img * (1 - a3) + np.array(colour)[None, None, :] * a3


def _arc(img, xx, yy, bx, by, r, soft, colour, ang_from, ang_to):
    """A hollow crescent, for peripapillary atrophy."""
    d = np.hypot(xx - bx, yy - by)
    th = np.arctan2(yy - by, xx - bx)
    th = np.where(th < 0, th + 2 * np.pi, th)

    a = np.zeros(d.shape)
    band = (d >= r * 0.92) & (d <= r + soft) & (th >= ang_from) & (th <= ang_to)
    a[band] = 1.0
    fade = band & (d > r)
    a[fade] = 1 - (d[fade] - r) / soft
    a = np.maximum(a, 0)
    a3 = a[:, :, None]
    return img * (1 - a3) + np.array(colour)[None, None, :] * a3
