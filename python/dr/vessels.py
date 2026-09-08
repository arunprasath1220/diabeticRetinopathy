"""The vessel-junction test.

The directional top-hat has one systematic blind spot, and it is the whole reason
a healthy retina once came back covered in microaneurysm marks.

The roundness test asks whether a structure is filled in by a linear closing in
*every* direction. A vessel is not - it survives the element lying along it. But
that argument only holds where a vessel is locally straight and alone. Where two
vessels cross, where one branches, and at the apex of a tight bend, there is no
direction in which the structure is straight, so every orientation fills it and it
seeds exactly as a lesion does. Those three configurations lie all over the
vascular arcades, and they are precisely the "veins" that were being reported as
lesions.

No refinement of the top-hat can separate them, because at the junction the two
really do look the same. What differs is the surroundings: a lesion is an isolated
blob with normal retina around it, while a junction has vessel running out of it in
three or four directions. So the region is judged by what leaves it.
"""

from __future__ import annotations

import numpy as np

from . import config as C


def vessel_arms(cx: float, cy: float, area: float,
                vessel_zone: np.ndarray, arm_length: float) -> dict:
    """Sample a ring just outside the region; report the arms leaving it.

    Contiguous runs of hit directions are merged, so one vessel several
    directions wide counts as one arm rather than several.
    """
    h, w = vessel_zone.shape
    r0 = np.sqrt(max(area, 1) / np.pi)
    inner = r0 + C.ARM_COLLAR_PAD
    outer = inner + max(8.0, arm_length)
    steps = max(6, int(round(outer - inner)))

    nd = C.ARM_DIRECTIONS
    hit = np.zeros(nd, dtype=bool)

    for d in range(nd):
        th = 2 * np.pi * d / nd
        c, s = np.cos(th), np.sin(th)
        # A perpendicular tolerance of one pixel keeps the ray on a vessel that
        # leans slightly, without letting it wander onto a neighbouring one.
        px, py = -s, c

        on = 0
        seen = 0
        for k in range(steps + 1):
            r = inner + (outer - inner) * k / steps
            any_ = False
            for o in (-1, 0, 1):
                x = int(round(cx + r * c + o * px))
                y = int(round(cy + r * s + o * py))
                if x < 0 or y < 0 or x >= w or y >= h:
                    continue
                if vessel_zone[y, x]:
                    any_ = True
                    break
            seen += 1
            on += any_
        if seen and on / seen >= C.ARM_PERSISTENCE:
            hit[d] = True

    hits = int(hit.sum())
    if hits == 0:
        return {"arms": 0, "coverage": 0.0, "angles": []}
    if hits == nd:
        return {"arms": 1, "coverage": 1.0, "angles": [0.0]}

    # Begin at a gap so that runs do not wrap around the end of the array.
    start = int(np.nonzero(~hit)[0][0])
    angles = []
    i = 0
    while i < nd:
        if not hit[(start + i) % nd]:
            i += 1
            continue
        length = 0
        total = 0
        while length < nd:
            j = (start + i + length) % nd
            if not hit[j]:
                break
            total += start + i + length
            length += 1
        angles.append(((total / length) % nd) * 360.0 / nd)
        i += length

    return {"arms": len(angles), "coverage": hits / nd, "angles": angles}


def is_through_vessel(ctx: dict) -> bool:
    """Two arms leaving in roughly opposite directions.

    A vessel runs *through* this region rather than ending, branching or turning
    in it. That case is kept, not rejected: a microaneurysm beside a venule looks
    exactly like this, and it is a real finding and by far the commonest early
    sign of diabetic retinopathy. Deleting the class would cost more than it saves.

    The tolerance is deliberately wide. A vessel curving gently past a small
    hemorrhage does not leave in exactly opposite directions, and the cost of the
    two errors is not symmetric: dropping a real lesion is worse than keeping a
    kink, so only a pronounced turn is called a bend.
    """
    if ctx["arms"] != 2:
        return False
    d = abs(ctx["angles"][0] - ctx["angles"][1])
    if d > 180:
        d = 360 - d
    return abs(d - 180) <= C.JUNCTION_ANTIPODAL_TOL_DEG


def is_vessel_junction(ctx: dict) -> bool:
    """Crossing, bifurcation or bend, rather than lesion?

    ==================  ==========================================================
    0 or 1 arm          free-standing lesion, or one sitting against a vessel
    2 opposite arms     a vessel passing straight through - KEPT, see above
    2 arms at an angle  a bend: the vessel turns here, and the turn was detected
    3 or more arms      a crossing or a bifurcation
    ==================  ==========================================================
    """
    if ctx["coverage"] >= C.JUNCTION_RING_SATURATION:
        return True
    if ctx["arms"] >= 3:
        return True
    if ctx["arms"] == 2:
        return not is_through_vessel(ctx)
    return False
