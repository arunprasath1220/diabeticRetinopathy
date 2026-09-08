"""Step 4: find and classify lesion candidates.

Classical morphology throughout, entirely independent of the CNN. That
independence is the point: Step 2 and Step 4 reach their conclusions by different
means, so when they disagree the disagreement is informative, and
:mod:`dr.grading` treats it as a reason to refer.

The method:

1. Mask and erode the retinal field, so the field-of-view edge - a huge intensity
   step - cannot generate candidates.
2. Seed from directional top-hats. No directional response anywhere in a region
   means no lesion, and that single test excludes the vessels, the macula and
   smooth background shading at once, without needing to know where any of them
   are.
3. Grow each seed into its full extent against a multi-scale background.
4. Reject on measured shape, boundedness, position and contrast.
5. Type what survives by size, shape and edge sharpness.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import config as C
from .field import aperture_geometry, retina_field_mask
from .imaging import (box_blur, dilate_mask, gradient_magnitude,
                      image_channels, masked_blur)
from .regions import region_features
from .seeds import round_structure_seeds
from .stats import robust_sigma
from .vessels import is_through_vessel, is_vessel_junction, vessel_arms


def detect_lesion_candidates(rgb: np.ndarray,
                             anatomy: Optional[dict] = None) -> dict:
    ch = image_channels(rgb)
    w, h, n = ch["w"], ch["h"], ch["n"]
    green, lum, red, blue = ch["green"], ch["lum"], ch["red"], ch["blue"]

    inside = retina_field_mask(lum, 0.05)
    retina_area = int(inside.sum())
    if retina_area < n * 0.05:
        raise ValueError("retinal area too small to analyse")

    green_s = box_blur(green, 1)
    lum_s = box_blur(lum, 1)
    grad = gradient_magnitude(lum_s)

    # ---- Lesion evidence -------------------------------------------------
    seeds = round_structure_seeds(green_s, lum_s, inside)
    dark_seed = seeds["darkSeed"]
    bright_seed = seeds["brightSeed"]
    vessel_mask = seeds["vesselMask"]
    vessel_zone = dilate_mask(vessel_mask, 1)

    yy, xx = np.mgrid[:h, :w]

    # Widest structure the ladder can still speak for. Past this the longest
    # element no longer fills the region, so no rung can show saturation, and
    # silence must not be read as a verdict - this is what keeps a genuinely large
    # blot hemorrhage from being deleted by a rule that was never about it.
    sat_max_area = np.pi * seeds["satSE"][-1] ** 2

    # Backgrounds exclude the vessels as well as the black surround. A vessel in
    # the averaging window drags the background down and makes ordinary retina
    # between two vessels read as abnormally bright.
    bg_mask = inside & ~vessel_zone

    # ---- Extent. Seeds say where lesions are; these say how far they reach.
    scales = [max(4, int(round(min(w, h) * 0.015))),
              max(10, int(round(min(w, h) * 0.055))),
              max(20, int(round(min(w, h) * 0.120)))]

    dark_loose = np.zeros((h, w), dtype=bool)
    bright_loose = np.zeros((h, w), dtype=bool)
    dark_dev = np.zeros((h, w))
    bright_dev = np.zeros((h, w))
    noise_dark = noise_bright = 1.0

    for si, r in enumerate(scales):
        bg_g = masked_blur(green_s, bg_mask, r)
        bg_l = masked_blur(lum_s, bg_mask, r)

        d_dark = bg_g - green_s        # dark lesions are dark in green
        d_bright = lum_s - bg_l        # bright lesions are bright in luminance

        s_dark = robust_sigma(d_dark, inside)
        s_bright = robust_sigma(d_bright, inside)
        if si == 0:
            noise_dark, noise_bright = s_dark, s_bright

        t_dark = max(C.DARK_FLOOR, s_dark * C.DARK_K) * C.HYST_RATIO
        t_bright = max(C.BRIGHT_FLOOR, s_bright * C.BRIGHT_K) * C.HYST_RATIO

        dark_loose |= inside & (d_dark > t_dark)
        bright_loose |= inside & (d_bright > t_bright)
        np.maximum(dark_dev, d_dark, out=dark_dev)
        np.maximum(bright_dev, d_bright, out=bright_dev)

    # Growth must not run along a vessel. A lesion lying on one is seeded
    # correctly, but the permissive mask covers the vessel as well, so the region
    # grows out along the whole vessel tree and is then thrown away as too large or
    # too elongated - taking the lesion with it.
    dark_loose[vessel_mask] = False
    bright_loose[vessel_mask] = False

    # ...and then every pixel that still carries lesion evidence is put back.
    # Restoring only the seeds is not enough once the vessel map is dense: a lesion
    # lying against a vessel has its seed restored but the rest of its body
    # deleted, so what is left to grow from is the sliver furthest from the vessel
    # - the lesion's own rim, which is the one shape that reads as unbounded, and
    # it is then thrown out by the saturation test. This cannot re-admit the
    # vessel, because it is the roundness response and a vessel barely produces
    # one: that is the whole basis on which the vessel map was built.
    dark_keep = seeds["tDark"] * C.HYST_RATIO
    bright_keep = seeds["tBright"] * C.HYST_RATIO
    dark_loose |= dark_seed | (seeds["roundDark"] > dark_keep)
    bright_loose |= bright_seed | (seeds["roundBright"] > bright_keep)
    dark_loose &= inside
    bright_loose &= inside

    sharp_edge_thresh = float(grad[inside].mean()) * 1.5

    max_area = round(n * 0.030)
    min_area = max(6, round(n * 0.000012))
    ma_max_area = max(min_area + 1, round(n * 0.00018))
    cws_min_area = round(n * C.CWS_MIN_AREA_FRAC)

    # ---- Position rules ---------------------------------------------------
    has_anatomy = anatomy is not None
    if has_anatomy:
        disc_r = anatomy["disc"]["radius"]
        disc_x = anatomy["disc"]["x"]
        disc_y = anatomy["disc"]["y"]
        disc_d2 = (xx - disc_x) ** 2 + (yy - disc_y) ** 2
    else:
        disc_r = 0.0
        disc_x = disc_y = 0.0
        disc_d2 = np.full((h, w), np.inf)
    has_macula = has_anatomy and anatomy.get("fovea") is not None

    # Measured disc tissue. The shape is checked rather than assumed: landmarks
    # and detection run on the same image today, but a mask indexed against a
    # different size would not fail loudly - it would quietly delete findings
    # somewhere else in the image, which is the worst way for this to go wrong.
    disc_tissue = None
    if has_anatomy and anatomy["disc"].get("tissue") is not None:
        t = anatomy["disc"]["tissue"]
        if t.shape == (h, w):
            disc_tissue = np.asarray(t, dtype=bool)

    # ---- Colour reference for the peripapillary ring ----------------------
    # Past the halo there is still nerve-fibre reflex and atrophic mottling, and
    # in shape, size and contrast those are indistinguishable from a hard exudate
    # - which is why every geometric rule tried here left some of them standing.
    # What separates them is what they are made of. An exudate is lipid and reads
    # yellow: it gains in red and green and hardly at all in blue. Disc tissue,
    # sclera and reflex are white or grey and gain in all three, so their blue
    # fraction climbs while an exudate's stays near the retina's.
    #
    # Both ends of the scale are read off this image, so nothing here depends on
    # the camera's white balance, and an image where the two ends do not separate
    # gets no colour test rather than a guessed one.
    chan_sum = red + green + blue
    blue_frac = np.full((h, w), -1.0)
    litm = chan_sum > 24                      # too dark to carry a readable hue
    blue_frac[litm] = blue[litm] / chan_sum[litm]

    disc_colour = None
    if has_anatomy and disc_r > 0:
        reach = disc_r * C.PERIPAPILLARY_MULT * 1.6
        readable = (disc_d2 <= reach ** 2) & inside & (blue_frac >= 0)

        if disc_tissue is None:
            white_ref = readable & (disc_d2 <= disc_r ** 2 * 0.64)   # inner core
        else:
            white_ref = readable & disc_tissue
        # Retina reference: outside the rim and off the vessels, so neither the
        # disc's own edge nor a vessel's blood colour sets the other end.
        retina_ref = readable & (disc_d2 > (disc_r * 1.6) ** 2) & ~vessel_zone
        if disc_tissue is not None:
            retina_ref &= ~disc_tissue

        if int(white_ref.sum()) >= 200 and int(retina_ref.sum()) >= 200:
            disc_f = float(np.median(blue_frac[white_ref]))
            retina_f = float(np.median(blue_frac[retina_ref]))
            if disc_f - retina_f >= C.DISC_COLOUR_MIN_SEPARATION:
                disc_colour = {
                    "discF": disc_f, "retinaF": retina_f,
                    "cut": retina_f + (disc_f - retina_f) * C.DISC_COLOUR_LEVEL,
                }

    ap = aperture_geometry(lum)
    edge_r2 = (ap["R"] * C.EDGE_ZONE_FRAC) ** 2
    edge_artifact_area = round(n * C.EDGE_ARTIFACT_AREA_FRAC)

    # The junction test is only meaningful at the scale vessels cross and branch
    # at. A large blot hemorrhage genuinely does have several vessels running out
    # of the area around it, and testing one would throw away a real and serious
    # finding, so anything wider than a couple of vessel widths is exempt.
    se_len = max(3, int(round(min(w, h) * C.LINEAR_SE_FRAC)))
    junction_max_area = round(np.pi * (se_len * 2.0) ** 2)
    junction_arm_len = se_len * 1.5

    rejected = {k: 0 for k in ("vessel", "junction", "tooLarge", "tooSmall",
                               "disc", "discTissue", "discColour", "weak",
                               "streak", "noSeed", "macula", "smooth",
                               "edge", "unbounded")}
    candidates: list = []
    dark_covered = np.zeros(h * w, dtype=bool)
    bright_covered = np.zeros(h * w, dtype=bool)

    # ------------------------------------------------------------------
    def sat_rung(area: float) -> int:
        """Which rung of the ladder speaks for a region of this size.

        The first element at least as wide as the region is across. Below that the
        region has not saturated yet and the ratio would only measure how much of
        it is still being filled in.
        """
        rr = np.sqrt(max(area, 1) / np.pi)
        for k in range(len(seeds["satSE"]) - 1):
            if seeds["satSE"][k] >= rr:
                return k
        return -1                              # wider than the longest element

    def growth_ratio(c: dict, dark: bool, k: int):
        """How much harder the response grows at the next element length.

        Measured over a disc centred on the region rather than over the region's
        own pixels, and only where the response clears a core level.

        Two reasons. The faint margins a region is grown out to carry almost no
        top-hat response at any element length, so including them drives both
        means toward zero and the ratio toward noise. More importantly the
        region's pixel list is not the structure: vessel pixels are removed from
        what a region may grow into, so a lesion touching a vessel is truncated,
        and what survives is the part furthest from the vessel - its own rim,
        which is exactly the shape that reads as unbounded. Measured that way the
        test rejected real lesions as soon as the vessel map became dense enough
        to bite: on one retina it took five of them.
        """
        maps = seeds["satDark"] if dark else seeds["satBright"]
        core = (seeds["tDark"] if dark else seeds["tBright"]) * 0.5
        small, big = maps[k], maps[k + 1]

        rad = max(3, int(round(np.sqrt(max(c["area"], 1) / np.pi))) + 1)
        x0 = max(0, int(round(c["cx"])) - rad)
        x1 = min(w - 1, int(round(c["cx"])) + rad)
        y0 = max(0, int(round(c["cy"])) - rad)
        y1 = min(h - 1, int(round(c["cy"])) + rad)

        sx = xx[y0:y1 + 1, x0:x1 + 1]
        sy = yy[y0:y1 + 1, x0:x1 + 1]
        disk = (sx - c["cx"]) ** 2 + (sy - c["cy"]) ** 2 <= rad ** 2

        s_small = small[y0:y1 + 1, x0:x1 + 1]

        # Vessel pixels are excluded from the measurement, not merely from the
        # region. A lesion beside a vessel is otherwise measured partly on the
        # vessel, and a vessel is the one thing that keeps responding harder as
        # the element grows - so the lesion inherits its neighbour's growth and is
        # rejected for it. Measured on one retina the affected lesions came out
        # between 1.08 and 1.37 against a cut of 1.05, while lesions in open
        # retina sit at 1.00. This is the vessel map protecting a finding rather
        # than suppressing one.
        sel = (disk & inside[y0:y1 + 1, x0:x1 + 1]
               & ~vessel_mask[y0:y1 + 1, x0:x1 + 1] & (s_small >= core))

        m = int(sel.sum())
        a = float(s_small[sel].sum())
        if m < 3 or a <= 0:
            return None                        # no responding core to judge
        return float(big[y0:y1 + 1, x0:x1 + 1][sel].sum()) / a

    def fails_saturation(c: dict, dark: bool) -> bool:
        """Bounded at any scale is bounded.

        A region wider than the longest element is exempt, because there the test
        is silent rather than negative.
        """
        if c["area"] > sat_max_area:
            return False
        k = sat_rung(c["area"])
        if k < 0:
            return False
        g = growth_ratio(c, dark, k)
        if g is None:
            return False
        return g > C.SATURATION_MAX_GROWTH

    def disc_tissue_frac(c: dict) -> float:
        if disc_tissue is None:
            return 0.0
        return float(disc_tissue.ravel()[c["pixels"]].sum()) / len(c["pixels"])

    def looks_like_disc_tissue(c: dict) -> bool:
        if disc_colour is None:
            return False
        v = blue_frac.ravel()[c["pixels"]]
        v = v[v >= 0]
        if v.size < 3:
            return False
        return float(v.mean()) >= disc_colour["cut"]

    def classify(comps: list, dark: bool) -> None:
        covered = dark_covered if dark else bright_covered
        for c in comps:
            # No directional response anywhere in the region means no lesion. This
            # one test excludes the vessels, the macula and smooth background
            # shading together, without needing to know where any of them are.
            if not c["hasSeed"]:
                rejected["noSeed"] += 1;   continue
            if c["area"] < min_area:
                rejected["tooSmall"] += 1; continue
            if c["area"] > max_area:
                rejected["tooLarge"] += 1; continue

            if has_anatomy:
                d2 = (c["cx"] - disc_x) ** 2 + (c["cy"] - disc_y) ** 2
            else:
                d2 = np.inf
            if d2 <= (disc_r * 1.15) ** 2:
                rejected["disc"] += 1; continue

            # Outside that circle the disc can still be the explanation for a
            # bright mark: the rim it was grown from, a crescent of sclera beside
            # it, the nerve-fibre reflex arcing off it. Two independent ways of
            # saying so, either sufficient - the mark is built out of measured
            # disc tissue, or it is the colour of disc tissue rather than of
            # exudate.
            #
            # Dark candidates are deliberately exempt. Nothing about the disc is
            # dark, so a dark mark beside it is a hemorrhage and has to survive;
            # and the disc's margin is where a disc hemorrhage sits, which is a
            # finding worth more than everything this rule removes.
            if not dark and d2 <= (disc_r * C.PERIPAPILLARY_MULT) ** 2:
                if disc_tissue_frac(c) >= C.DISC_TISSUE_OVERLAP:
                    rejected["discTissue"] += 1; continue
                if looks_like_disc_tissue(c):
                    rejected["discColour"] += 1; continue

            # A large region far out toward the aperture rim is an artifact, not a
            # lesion: a shadow from pupil misalignment, or peripapillary
            # vignetting. Genuine peripheral lesions are small, so the rule is
            # keyed on size as well as position and small ones stay.
            if c["area"] >= edge_artifact_area:
                de2 = (c["cx"] - ap["cx"]) ** 2 + (c["cy"] - ap["cy"]) ** 2
                if de2 > edge_r2:
                    rejected["edge"] += 1; continue

            if c["aspect"] > 4.0 and c["fillRatio"] < 0.30:
                rejected["streak"] += 1; continue

            # A region lying almost entirely on the vessel map is a piece of
            # vessel, whatever seeded it. The threshold is high on purpose: a
            # lesion touching a vessel overlaps it partially and must survive.
            on_vessel_frac = (float(vessel_zone.ravel()[c["pixels"]].sum())
                              / len(c["pixels"]))
            if on_vessel_frac > C.VESSEL_OVERLAP_REJECT:
                rejected["vessel"] += 1; continue

            # Crossings, bifurcations, bends and choroidal mottling answer the
            # roundness test at one element length exactly as a lesion does. They
            # are told apart by whether the response stops growing when the
            # element does: a lesion is bounded, they are not.
            if fails_saturation(c, dark):
                rejected["unbounded"] += 1; continue

            on_vessel = False
            if dark and c["area"] <= junction_max_area:
                ctx = vessel_arms(c["cx"], c["cy"], c["area"],
                                  vessel_zone, junction_arm_len)
                if is_vessel_junction(ctx):
                    rejected["junction"] += 1; continue
                on_vessel = is_through_vessel(ctx) or ctx["arms"] >= 1

            floor_v = C.DARK_CONTRAST_FLOOR if dark else C.BRIGHT_CONTRAST_FLOOR
            noise = noise_dark if dark else noise_bright
            # Contrast is measured against a background that excludes vessels, so
            # a vessel scores well on it simply for being a vessel. A candidate on
            # the vasculature therefore has to clear a higher bar than one in open
            # retina before the same figure means the same thing.
            bar = max(floor_v, noise * C.MIN_CONTRAST_K)
            if on_vessel:
                bar *= C.ON_VESSEL_CONTRAST_FACTOR
            if c["maxContrast"] < bar:
                rejected["weak"] += 1; continue

            if dark:
                if (c["area"] <= ma_max_area and c["aspect"] <= 2.1
                        and c["fillRatio"] >= 0.45):
                    type_key = "MA"
                else:
                    type_key = "HEM"
            elif c["meanGradient"] >= sharp_edge_thresh:
                type_key = "HE"
            elif c["area"] >= cws_min_area:
                type_key = "CWS"
            else:
                rejected["weak"] += 1; continue

            rec = dict(c)
            rec["type"] = type_key
            rec["onVessel"] = bool(on_vessel)
            candidates.append(rec)
            covered[c["pixels"]] = True

    # ---- Pass 1: grown regions -------------------------------------------
    classify(region_features(dark_loose, dark_dev, grad, dark_seed), True)
    classify(region_features(bright_loose, bright_dev, grad, bright_seed), False)

    # ---- Pass 2: safety net -----------------------------------------------
    # Something that was seeded must not vanish because the region it grew into
    # failed a size or shape test; the evidence for the lesion was the seed, not
    # the extent. Any seed not covered by an accepted region is reconsidered on
    # its own extent, so the worst case is that a lesion is reported slightly
    # smaller than it really is, never that it goes unreported. Seeds cannot
    # reintroduce vessels, because vessels never seed.
    dark_left = dark_seed & ~dark_covered.reshape(h, w)
    bright_left = bright_seed & ~bright_covered.reshape(h, w)
    if dark_left.any():
        classify(region_features(dark_left, dark_dev, grad, None), True)
    if bright_left.any():
        classify(region_features(bright_left, bright_dev, grad, None), False)

    candidates.sort(key=lambda c: -c["area"])

    counts = {k: 0 for k in C.LESION_ORDER}
    lesion_pixels = 0
    on_vessel_count = 0
    for c in candidates:
        counts[c["type"]] += 1
        lesion_pixels += c["area"]
        on_vessel_count += int(c["onVessel"])

    return {
        "candidates": candidates, "counts": counts, "rejected": rejected,
        "w": w, "h": h, "retinaArea": retina_area,
        "lesionPixels": lesion_pixels, "onVesselCount": on_vessel_count,
        "noise": {"dark": noise_dark, "bright": noise_bright},
        "scales": scales,
        "discExcluded": has_anatomy,
        "maculaExcluded": has_macula,
        "discExtentMeasured": bool(has_anatomy
                                   and anatomy["disc"].get("measuredExtent")),
        "discTissueMask": disc_tissue is not None,
        "discColourUsed": disc_colour is not None,
    }
