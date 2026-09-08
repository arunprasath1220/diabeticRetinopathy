"""Connected components with the shape and contrast features the type rules need.

Four-connectivity, not eight. These are compact blobs, and eight-connectivity
bridges two lesions that merely touch at a corner into one region, which changes
both the count and the shape tests applied to it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage as ndi

_FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def region_features(bw: np.ndarray, contrast_map: np.ndarray,
                    gradient_map: np.ndarray,
                    seed_mask: Optional[np.ndarray] = None) -> list:
    """Label ``bw`` and measure every component.

    Each returned dict carries:

    ``pixels``
        flat indices, kept rather than discarded because three later tests -
        vessel overlap, disc-tissue overlap and mean hue - are all "what fraction
        of this region lies on that map", and none of them can be answered from
        summary geometry.
    ``fillRatio``, ``aspect``
        how solidly the bounding box is filled, and how elongated it is. Together
        they reject streaks without rejecting the small round things.
    ``maxContrast``
        the strongest deviation anywhere in the region, not the mean. A lesion
        with a faint margin still has a core, and averaging over the margin
        buries it.
    ``meanGradient``
        edge sharpness, which is what separates hard exudate from cotton wool.
    ``hasSeed``
        whether any pixel carries directional evidence. A region without one is
        not a lesion however it looks.

    The per-label aggregation is done with sorted flat indices rather than a loop
    over components, because on a noisy image there can be tens of thousands of
    them and a Python loop over that is the slowest thing in the pipeline.
    """
    lab, n = ndi.label(np.asarray(bw, dtype=bool), structure=_FOUR)
    if n == 0:
        return []

    h, w = lab.shape
    flat = lab.ravel()

    # Group the flat indices by label in one pass.
    order = np.argsort(flat, kind="stable")
    sorted_labels = flat[order]
    counts = np.bincount(flat, minlength=n + 1)
    ends = np.cumsum(counts)
    starts = ends - counts

    ys, xs = np.divmod(order, w)

    idx = np.arange(1, n + 1)
    areas = counts[1:]
    sum_x = np.bincount(flat, weights=np.tile(np.arange(w), h).astype(np.float64),
                        minlength=n + 1)[1:]
    sum_y = np.bincount(flat, weights=np.repeat(np.arange(h), w).astype(np.float64),
                        minlength=n + 1)[1:]

    min_x = ndi.minimum(np.tile(np.arange(w), h).reshape(h, w), lab, idx)
    max_x = ndi.maximum(np.tile(np.arange(w), h).reshape(h, w), lab, idx)
    min_y = ndi.minimum(np.repeat(np.arange(h), w).reshape(h, w), lab, idx)
    max_y = ndi.maximum(np.repeat(np.arange(h), w).reshape(h, w), lab, idx)

    max_contrast = ndi.maximum(contrast_map, lab, idx)
    mean_contrast = ndi.mean(contrast_map, lab, idx)
    mean_gradient = ndi.mean(gradient_map, lab, idx)

    if seed_mask is None:
        has_seed = np.ones(n, dtype=bool)
    else:
        has_seed = np.asarray(
            ndi.maximum(np.asarray(seed_mask, dtype=np.float64), lab, idx)) > 0

    min_x = np.atleast_1d(min_x)
    max_x = np.atleast_1d(max_x)
    min_y = np.atleast_1d(min_y)
    max_y = np.atleast_1d(max_y)
    max_contrast = np.atleast_1d(max_contrast)
    mean_contrast = np.atleast_1d(mean_contrast)
    mean_gradient = np.atleast_1d(mean_gradient)
    has_seed = np.atleast_1d(has_seed)

    comps = []
    for k in range(n):
        label = k + 1
        area = int(areas[k])
        bw_box = int(max_x[k] - min_x[k]) + 1
        bh_box = int(max_y[k] - min_y[k]) + 1
        comps.append({
            "pixels": order[starts[label]:ends[label]],
            "area": area,
            "cx": float(sum_x[k] / area),
            "cy": float(sum_y[k] / area),
            "bw": bw_box,
            "bh": bh_box,
            "fillRatio": area / max(1, bw_box * bh_box),
            "aspect": max(bw_box, bh_box) / max(1, min(bw_box, bh_box)),
            "maxContrast": float(max_contrast[k]),
            "meanContrast": float(mean_contrast[k]),
            "meanGradient": float(mean_gradient[k]),
            "hasSeed": bool(has_seed[k]),
        })
    return comps
