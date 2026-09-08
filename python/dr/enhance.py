"""Step 1b: denoise, flatten illumination, restore local contrast.

Works on luminance alone and puts the original chroma back afterwards, so nothing
here can shift the colour of a lesion. That matters downstream: the peripapillary
test in :mod:`dr.lesions` separates yellow exudate from white disc tissue by hue,
and an enhancement that touched chroma would quietly break it.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .imaging import box_blur


def enhance_image(rgb: np.ndarray) -> np.ndarray:
    """Three stages, in this order.

    1. A light denoise - 30% of a radius-1 box blur - so that the illumination
       estimate below is not fitted to sensor noise.
    2. Illumination flattening. Subtracting a heavily blurred copy and adding the
       global mean back removes the vignette and the uneven flash falloff that
       every handheld fundus camera produces, without changing the mean level the
       later thresholds are calibrated against.
    3. CLAHE, which is what makes faint lesions visible again after the flattening
       has compressed the range.

    The colour conversion is written out rather than delegated, because the
    library conversions implement the studio-swing BT.601 convention where luma
    occupies 16..235. Every threshold in this pipeline is stated on a full-range
    0..255 scale, and routing through the studio-swing form would silently
    rescale all of them.
    """
    d = np.asarray(rgb, dtype=np.float64)
    r, g, b = d[:, :, 0], d[:, :, 1], d[:, :, 2]

    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    Cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0

    # 1. light denoise
    mix = 0.3
    Y = Y * (1 - mix) + box_blur(Y, 1) * mix

    # 2. illumination flattening
    h, w = Y.shape
    radius = max(4, int(round(min(w, h) / 10)))
    Y = np.clip(Y - box_blur(Y, radius) + Y.mean(), 0, 255)

    # 3. local contrast
    Y = clahe(Y, 8, 8)

    out_r = Y + 1.402 * (Cr - 128.0)
    out_g = Y - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0)
    out_b = Y + 1.772 * (Cb - 128.0)

    out = np.stack([out_r, out_g, out_b], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe(Y: np.ndarray, tiles_x: int = 8, tiles_y: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation, written out in full.

    ``cv2.createCLAHE`` would do this in one call and its clip limit even uses the
    same convention (a multiple of the tile's mean bin count). It is not used
    here, and the reason is worth recording: OpenCV pads the image so that every
    tile is the same size, while this pipeline's reference implementation lets the
    last row and column of tiles be smaller. That changes the mapping near two
    edges of the frame by a grey level or two, which is invisible to look at and
    is exactly the kind of difference that turns into an unexplainable
    disagreement in a lesion count three stages later.

    So the tiling, the clip limit, the single-pass redistribution and the bilinear
    interpolation between tile centres are all reproduced.
    """
    Y = np.asarray(Y, dtype=np.float64)
    h, w = Y.shape

    tile_w = int(np.ceil(w / tiles_x))
    tile_h = int(np.ceil(h / tiles_y))

    q = np.clip(np.round(Y), 0, 255).astype(np.int64)

    # ---- One mapping per tile -------------------------------------------
    maps = np.zeros((tiles_y, tiles_x, 256), dtype=np.float64)
    for ty in range(tiles_y):
        y0, y1 = ty * tile_h, min(h, (ty + 1) * tile_h)
        for tx in range(tiles_x):
            x0, x1 = tx * tile_w, min(w, (tx + 1) * tile_w)
            block = q[y0:y1, x0:x1]
            count = block.size or 1

            hist = np.bincount(block.ravel(), minlength=256).astype(np.float64)

            # Clip the histogram and give the excess back uniformly. Without the
            # clip, a tile that is mostly flat background gets an enormous slope
            # at its dominant level and amplifies its own noise into texture.
            clip_limit = (count / 256.0) * 3.5
            excess = np.maximum(hist - clip_limit, 0).sum()
            hist = np.minimum(hist, clip_limit) + excess / 256.0

            maps[ty, tx] = np.clip(np.cumsum(hist) / count * 255.0, 0, 255)

    # ---- Bilinear interpolation between the four nearest tile centres ----
    xs = np.arange(w)
    ys = np.arange(h)
    fx = (xs - tile_w / 2.0) / tile_w
    fy = (ys - tile_h / 2.0) / tile_h

    tx0 = np.floor(fx).astype(np.int64)
    ty0 = np.floor(fy).astype(np.int64)
    wx = fx - tx0
    wy = fy - ty0

    tx1 = np.clip(tx0 + 1, 0, tiles_x - 1)
    ty1 = np.clip(ty0 + 1, 0, tiles_y - 1)
    tx0 = np.clip(tx0, 0, tiles_x - 1)
    ty0 = np.clip(ty0, 0, tiles_y - 1)

    TY0 = ty0[:, None]
    TY1 = ty1[:, None]
    TX0 = tx0[None, :]
    TX1 = tx1[None, :]
    WX = wx[None, :]
    WY = wy[:, None]

    m00 = maps[TY0, TX0, q]
    m10 = maps[TY0, TX1, q]
    m01 = maps[TY1, TX0, q]
    m11 = maps[TY1, TX1, q]

    top = m00 * (1 - WX) + m10 * WX
    bot = m01 * (1 - WX) + m11 * WX
    return top * (1 - WY) + bot * WY
