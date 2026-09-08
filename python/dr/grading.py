"""Step 5: apply the ICDR rule to what Step 4 found.

The published scale is defined by which lesion types are present and, for severe
disease, by how many hemorrhages appear in each quadrant:

===  ==========================  =============================================
0    No apparent retinopathy     no abnormality
1    Mild NPDR                   microaneurysms only
2    Moderate NPDR               more than microaneurysms, less than severe
3    Severe NPDR                 no PDR signs, and any of the 4-2-1 rule:
                                 >20 intraretinal hemorrhages in each of 4
                                 quadrants, venous beading in 2 or more, or
                                 prominent IRMA in 1 or more
4    Proliferative DR            neovascularisation, or vitreous/preretinal
                                 hemorrhage
===  ==========================  =============================================

Applying it here is legitimate because the rule is itself a function of lesion
type and count, which is exactly what Step 4 produces. What it is NOT is a
diagnosis: the counts are unvalidated candidates, so the grade inherits every one
of their errors.

The hard ceiling
----------------
Two of the three arms of the 4-2-1 rule are venous beading and IRMA, and level 4
is defined by neovascularisation. This detector cannot see any of the three,
because all are elongated structures that its vessel test removes by construction.
So it can reach level 3 only through the hemorrhage arm, can never reach level 4,
and - the part that matters clinically - can never rule either of them out. That
limitation is returned with every result rather than buried, because a low grade
here does not mean a low grade in the eye.
"""

from __future__ import annotations

from typing import Optional

from . import config as C
from .anatomy import quadrant_label

CEILING = ("cannot detect venous beading, IRMA or neovascularisation, so level 4 "
           "is unreachable and neither it nor severe disease can be ruled out")


def quadrant_lesion_counts(lesions: Optional[dict],
                           anatomy: Optional[dict]) -> Optional[dict]:
    """Candidate counts per type per retinal quadrant.

    Returns ``None`` when the landmarks are unavailable - because without a disc
    position there are no quadrant axes, and inventing them would put every count
    in the wrong cell.
    """
    if not anatomy or not lesions:
        return None

    grid = {k: {q: 0 for q in C.QUADRANT_NAMES} for k in C.LESION_ORDER}
    for c in lesions["candidates"]:
        q = quadrant_label(c["cx"], c["cy"], anatomy["disc"], anatomy["nasalSide"])
        if c["type"] in grid and q in grid[c["type"]]:
            grid[c["type"]][q] += 1
    return grid


def assess_severity(lesions: Optional[dict], anatomy: Optional[dict],
                    grading: Optional[dict], quality: Optional[dict]) -> dict:
    counts = lesions["counts"] if lesions else None
    ma = counts["MA"] if counts else 0
    hem = counts["HEM"] if counts else 0
    he = counts["HE"] if counts else 0
    cws = counts["CWS"] if counts else 0
    beyond_ma = hem + he + cws
    total = ma + beyond_ma

    grid = quadrant_lesion_counts(lesions, anatomy)
    quads_over = 0
    if grid:
        quads_over = sum(1 for q in C.QUADRANT_NAMES
                         if grid["HEM"][q] > C.ICDR_HEM_PER_QUADRANT)
    hem_arm_met = grid is not None and quads_over == 4

    basis = []
    if lesions is None:
        level, label = None, "Not assessed"
        basis.append("lesion detection did not run on this image")
    elif hem_arm_met:
        level, label = 3, "Severe NPDR pattern"
        basis.append(f"more than {C.ICDR_HEM_PER_QUADRANT} hemorrhage candidates "
                     "in each of the four quadrants, which is the hemorrhage arm "
                     "of the 4-2-1 rule")
    elif beyond_ma > 0:
        level, label = 2, "Moderate NPDR pattern"
        basis.append(f"more than microaneurysms alone: {hem} hemorrhage, {he} hard "
                     f"exudate and {cws} cotton wool candidates")
    elif ma > 0:
        level, label = 1, "Mild NPDR pattern"
        basis.append(f"{ma} microaneurysm candidate{'' if ma == 1 else 's'} and "
                     "nothing else")
    else:
        level, label = 0, "No retinopathy observed"
        basis.append("no lesion candidate passed the detector on this image")

    # ---- Everything that makes this estimate untrustworthy here ----------
    doubts = []
    if lesions is None:
        doubts.append("lesion detection did not complete")
    if anatomy is None:
        doubts.append("landmarks could not be estimated, so the quadrant rule for "
                      "severe disease could not be applied at all")
    if quality and quality.get("verdict") == "enhance":
        doubts.append("image quality was borderline and had to be enhanced before "
                      "analysis")
    if lesions and total > C.NOISE_SUSPICION_COUNT:
        doubts.append("the candidate count is high enough to suggest the detector "
                      "is responding to image noise")
    if grid and 0 < quads_over < 4:
        doubts.append(f"hemorrhage candidates exceed the severe-disease threshold "
                      f"in {quads_over} of four quadrants, which sits right on the "
                      "boundary of the rule")

    # ---- The cross-check the two independent stages exist to provide -----
    if grading:
        model_says_dr = grading["predClass"] == 1
        if model_says_dr and level == 0:
            doubts.append("the classifier reports disease present while the "
                          "detector found no lesion at all, and the two disagree")
        if not model_says_dr and level is not None and level >= 2:
            doubts.append("the classifier reports no disease while the detector "
                          "found lesions beyond microaneurysms, and the two "
                          "disagree")
    else:
        doubts.append("the classifier did not run, so there is no independent "
                      "check on this result")

    # Refer whenever the rule says referable, whenever anything is in doubt, and
    # whenever any lesion at all was seen. Only a clean, agreeing, lesion-free
    # image avoids it. In a screening programme the cost of the two errors is not
    # symmetric, and this is where that asymmetry is written down.
    referable = level is not None and level >= 2
    uncertain = len(doubts) > 0
    refer = referable or uncertain or (level is not None and level >= 1)

    if referable:
        reason = ("This image reaches the referable threshold (moderate NPDR or "
                  "worse) under the ICDR rule.")
    elif uncertain:
        reason = "This result is not reliable enough to stand on its own."
    elif level is not None and level >= 1:
        reason = "Lesions were seen. Any retinopathy needs a specialist opinion."
    else:
        reason = ("Severe disease and proliferative disease cannot be excluded by "
                  "this method.")

    return {"level": level, "label": label, "basis": basis, "doubts": doubts,
            "refer": refer, "referable": referable, "uncertain": uncertain,
            "reason": reason, "grid": grid, "quadsOverThreshold": quads_over,
            "counts": counts, "total": total, "ceiling": CEILING}
