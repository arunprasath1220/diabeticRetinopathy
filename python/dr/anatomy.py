"""Step 3: the optic disc, the fovea, and the quadrant axes they define."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from . import config as C
from .field import retina_field_mask
from .imaging import image_channels, masked_blur, normalize_inside
from .morphology import flood_bright_region, square_closing


def estimate_optic_disc(rgb: np.ndarray) -> dict:
    """Locate the optic disc and measure its extent.

    Finding it. Brightness alone picks the wrong region whenever the photo carries
    a specular highlight or a blown-out patch, so brightness is combined with
    local contrast: the disc carries the vessel trunk and a sharp rim and is
    therefore bright AND textured, while flare is bright and smooth.

    Measuring it. This is where a real bug lived. The radius used to come from the
    area of the plateau surviving a cut at a fixed height above the *global*
    retinal mean, and both halves of that are wrong. The reference is global, so a
    change in average brightness anywhere else in the frame moves the disc's
    boundary; and the height is fixed, so on a soft-edged structure the contour
    sits inside the rim rather than on it. Both push the same way, and the
    measured radius came out a few per cent small.

    A few per cent sounds harmless until it is multiplied by the 1.15 exclusion
    factor and asked to reach past the rim, which it does not. The rim itself, a
    scleral crescent beside it and the peripapillary reflex all fall in the gap
    between where the circle stops and where the disc actually ends - and every
    one of them is bright, round and sharply bounded, which is the description of
    a hard exudate. They were duly reported as one.

    So the extent is grown instead, from a level referred to the retina
    immediately around the disc, and placed at the half-maximum point where the
    contour of a blurred edge actually sits. Twice: once for the disc proper, and
    once lower for the halo it shades into.
    """
    ch = image_channels(rgb)
    w, h, lum = ch["w"], ch["h"], ch["lum"]

    inside = retina_field_mask(lum, 0.03)
    if not inside.any():
        raise ValueError("no usable retinal area for disc estimation")

    # ---- Seed: bright AND textured --------------------------------------
    r = max(5, int(round(min(w, h) * 0.045)))
    blur = masked_blur(lum, inside, r)
    blur_sq = masked_blur(lum ** 2, inside, r)
    sd = np.sqrt(np.maximum(0.0, blur_sq - blur ** 2))

    nb = normalize_inside(blur, inside)
    ns = normalize_inside(sd, inside)

    score = nb * 0.65 + ns * 0.35
    score = np.where(inside, score, -np.inf)
    best_idx = int(np.argmax(score))
    seed_y, seed_x = divmod(best_idx, w)
    best = float(score.ravel()[best_idx])

    # ---- The old plateau estimate, kept as the fallback ------------------
    mean_blur = float(blur[inside].mean())
    peak = float(blur[seed_y, seed_x])
    cut = peak - (peak - mean_blur) * 0.35

    box = int(round(min(w, h) * 0.16))
    sub = np.zeros((h, w), dtype=bool)
    sub[max(0, seed_y - box):min(h, seed_y + box),
        max(0, seed_x - box):min(w, seed_x + box)] = True
    plateau = sub & inside & (blur >= cut)

    r_min = min(w, h) * 0.035
    r_max = min(w, h) * 0.11

    area = int(plateau.sum())
    if area > 0:
        ys, xs = np.nonzero(plateau)
        disc_x, disc_y = float(xs.mean()), float(ys.mean())
    else:
        disc_x, disc_y = float(seed_x), float(seed_y)
    radius = float(np.clip(np.sqrt(max(area, 1) / np.pi), r_min, r_max))

    # ---- Measured extent --------------------------------------------------
    cx0 = int(np.clip(round(disc_x), 0, w - 1))
    cy0 = int(np.clip(round(disc_y), 0, h - 1))
    r0 = radius

    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx0) ** 2 + (yy - cy0) ** 2

    # Peripapillary background: retina in an annulus outside the disc, close
    # enough to share its illumination. The inner edge is held clear of the blur
    # radius as well as of the disc, because the same blur that makes the disc a
    # plateau also spreads it outward, and an annulus inside that spill would read
    # the disc back as its own background and shrink the fill to nothing.
    ring_inner = max(r0 * 2.5, r0 + r * 1.5)
    ring_outer = max(ring_inner * 1.6, r0 * 4.0)
    ring = inside & (d2 >= ring_inner ** 2) & (d2 <= ring_outer ** 2)

    if int(ring.sum()) > 200:
        pp_bg = float(np.median(blur[ring]))
    else:
        # The disc sits near the edge of the aperture and no annulus fits. The
        # global mean is a worse reference, but it errs toward a smaller fill
        # rather than a runaway one, which is the safe direction.
        pp_bg = mean_blur

    # The disc's own level, as a high quantile of its core rather than the single
    # brightest pixel in it, so one specular speck on the cup cannot set the scale
    # that both cuts are measured against.
    core = inside & (d2 <= max(2.0, r0 * 0.6) ** 2)
    disc_level = float(np.percentile(blur[core], 75)) if core.any() else peak

    tissue: Optional[np.ndarray] = None
    measured_extent = False
    span = disc_level - pp_bg

    # A span of a grey level or less means the disc is not separable from the
    # retina around it on this frame. Anything grown from that is noise, so
    # nothing is grown and the plateau estimate is left to stand alone.
    if span > 1:
        max_r = r0 * C.DISC_FILL_MAX_MULT
        max_area = np.pi * (r0 * 2.8) ** 2

        edge = flood_bright_region(blur, inside, cx0, cy0,
                                   pp_bg + span * C.DISC_EDGE_LEVEL,
                                   max_r, max_area)
        if edge is not None:
            radius = float(np.clip(np.sqrt(edge["area"] / np.pi), r_min, r_max))
            disc_x, disc_y = edge["cx"], edge["cy"]
            tissue = edge["mask"]
            measured_extent = True

        # The halo takes in whatever the disc shades into: the rim it was grown
        # from, a scleral crescent beside it, the nerve-fibre reflex arcing off
        # it. It supersedes the edge mask when it can be grown, being the larger
        # of the two and the one the bright-candidate test actually wants.
        halo = flood_bright_region(blur, inside, cx0, cy0,
                                   pp_bg + span * C.DISC_HALO_LEVEL,
                                   max_r, max_area)
        if halo is not None:
            tissue = halo["mask"]

    return {"x": disc_x, "y": disc_y, "radius": radius, "score": best,
            "tissue": tissue, "measuredExtent": measured_extent}


def estimate_fovea_macula(rgb: np.ndarray, disc: dict) -> dict:
    """Locate the fovea, and with it the nasal side of the frame.

    Pure geometry from the disc puts the marker in roughly the right area but
    rarely on the fovea itself, so the position is measured: the fovea is the
    darkest part of the central retina, being avascular and pigment-dense. The
    vessels are removed first by a grey closing at vessel scale, because otherwise
    the darkest thing in the macula is a vein.

    The search is anchored as well as measured. Darkness alone let any dark
    structure in the window capture the marker; weighting it by distance from the
    anatomically expected point - about two and a half disc diameters temporal, a
    little below the disc's level - keeps the estimate where the fovea has to be
    while still letting real macular darkening move it. The 0.4 floor on the
    darkness term is what makes a featureless macula degrade to the anatomical
    position instead of snapping to noise.
    """
    src = np.asarray(rgb)
    H, W = src.shape[:2]

    # The macula is a broad feature, so the search runs on a downscaled copy.
    # That makes the morphology cheap and stops single dark pixels mattering.
    scale = min(1.0, 256 / max(H, W))
    if scale < 1.0:
        small = cv2.resize(src, (max(1, int(round(W * scale))),
                                 max(1, int(round(H * scale)))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = src
    sf = small.shape[1] / W

    ch = image_channels(small)
    w, h, lum = ch["w"], ch["h"], ch["lum"]
    inside = retina_field_mask(lum, 0.06)

    dx = disc["x"] * sf
    dy = disc["y"] * sf
    dd = max(6.0, disc["radius"] * 2 * sf)
    sgn = 1 if (W / 2 - disc["x"]) >= 0 else -1

    vr = max(2, int(round(dd * 0.16)))
    vessel_free = square_closing(lum, vr)

    bg = masked_blur(vessel_free, inside, max(6, int(round(dd * 1.5))))
    darkness = np.maximum(0.0, bg - vessel_free)
    darkness[~inside] = 0
    d_max = float(darkness.max()) or 1.0

    ex = dx + sgn * dd * 2.5
    ey = dy + dd * 0.3
    sigma = dd * 0.8
    two_sig2 = 2 * sigma ** 2
    reach = int(round(sigma * 2))

    yy, xx = np.mgrid[:h, :w]
    prior = np.exp(-((xx - ex) ** 2 + (yy - ey) ** 2) / two_sig2)
    score = (0.4 + 0.6 * (darkness / d_max)) * prior

    window = np.zeros((h, w), dtype=bool)
    window[max(0, int(round(ey - reach))):min(h, int(round(ey + reach))),
           max(0, int(round(ex - reach))):min(w, int(round(ex + reach)))] = True
    window &= inside

    if window.any():
        masked = np.where(window, score, -np.inf)
        bi = int(np.argmax(masked))
        by, bx = divmod(bi, w)
        best = float(masked.ravel()[bi])

        # Score-weighted centroid, so the marker sits in the middle of the dark
        # area rather than on its single darkest pixel.
        box = int(round(dd * 0.5))
        near = np.zeros((h, w), dtype=bool)
        near[max(0, by - box):min(h, by + box),
             max(0, bx - box):min(w, bx + box)] = True
        sel = near & inside & (score >= best * 0.75)

        if sel.any():
            wt = score[sel]
            fx = float((xx[sel] * wt).sum() / wt.sum())
            fy = float((yy[sel] * wt).sum() / wt.sum())
        else:
            fx, fy = float(bx), float(by)

        fovea = {"x": fx / sf, "y": fy / sf}
        evidence = ("measured — darkest macular region after vessel removal, "
                    "anchored to the expected position relative to the disc")
    else:
        fovea = {"x": ex / sf, "y": ey / sf}
        evidence = ("geometry only — no usable retinal area in the expected "
                    "zone, so the anatomical position is used unrefined")

    margin = disc["radius"] * 1.2
    fovea["x"] = float(np.clip(fovea["x"], margin, W - margin))
    fovea["y"] = float(np.clip(fovea["y"], margin, H - margin))

    # The disc is nasal to the fovea in either eye, so the side the disc sits on
    # is the nasal side - no need to know OD from OS.
    nasal_side = "left" if (fovea["x"] - disc["x"]) >= 0 else "right"

    return {"fovea": fovea,
            "maculaRadius": disc["radius"] * 2 * 1.1,
            "nasalSide": nasal_side,
            "dir": {"x": sgn, "y": 0},
            "evidence": evidence}


def quadrant_label(px: float, py: float, disc: dict, nasal_side: str) -> str:
    """Name the retinal quadrant a point falls in.

    The axes run through the optic disc, which is what the ICDR 4-2-1 rule counts
    hemorrhages against. Nasal and temporal are assigned from which side of the
    disc the fovea lies on, so the labelling is correct for a right or a left eye
    without ever being told which it is.
    """
    vertical = "Superior" if py < disc["y"] else "Inferior"
    left_of_disc = px < disc["x"]
    if nasal_side == "left":
        horizontal = "Nasal" if left_of_disc else "Temporal"
    else:
        horizontal = "Temporal" if left_of_disc else "Nasal"
    return f"{vertical}-{horizontal}"


def estimate_anatomy(rgb: np.ndarray) -> dict:
    """Both landmarks together, in the form the rest of the pipeline expects."""
    disc = estimate_optic_disc(rgb)
    fm = estimate_fovea_macula(rgb, disc)
    return {"disc": disc, "fovea": fm["fovea"],
            "maculaRadius": fm["maculaRadius"], "nasalSide": fm["nasalSide"],
            "dir": fm["dir"], "evidence": fm["evidence"]}
