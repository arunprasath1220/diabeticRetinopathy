"""Panel rendering: the images the UI displays.

Rendered server-side rather than in the browser so that the picture and the
numbers come from the same array. When the overlay is drawn from one copy of the
data and the table filled from another, they drift, and a mark that appears in one
and not the other is the hardest kind of bug to notice.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from . import config as C

# Turbo-like ramp, ordered so lightness rises with the value. That keeps the tones
# readable in the order they mean something, which a plain rainbow does not.
_CAM_STOPS = [
    (0.00, (48, 18, 59)), (0.15, (60, 100, 200)), (0.30, (30, 170, 220)),
    (0.45, (40, 210, 150)), (0.60, (150, 225, 60)), (0.75, (250, 205, 40)),
    (0.90, (240, 110, 30)), (1.00, (160, 20, 20)),
]

_TYPE_COLOUR = {          # BGR, because that is what OpenCV draws in
    "MA":  (60, 220, 255),
    "HEM": (60, 60, 240),
    "HE":  (80, 220, 80),
    "CWS": (240, 120, 240),
}


def _cam_lut() -> np.ndarray:
    """256-entry RGB lookup table for the activation ramp."""
    lut = np.zeros((256, 3), np.uint8)
    stops = _CAM_STOPS
    for i in range(256):
        t = i / 255.0
        if t <= stops[0][0]:
            rgb = stops[0][1]
        elif t >= stops[-1][0]:
            rgb = stops[-1][1]
        else:
            rgb = stops[-1][1]
            for k in range(1, len(stops)):
                if t <= stops[k][0]:
                    t0, c0 = stops[k - 1]
                    t1, c1 = stops[k]
                    f = (t - t0) / (t1 - t0)
                    rgb = tuple(c0[j] + (c1[j] - c0[j]) * f for j in range(3))
                    break
        lut[i] = np.array(rgb, np.uint8)
    return lut


_LUT = _cam_lut()


def binary_mask(lesions: dict) -> np.ndarray:
    """Every candidate pixel, white on black."""
    m = np.zeros((lesions["h"], lesions["w"]), np.uint8)
    flat = m.ravel()
    for c in lesions["candidates"]:
        flat[c["pixels"]] = 255
    return m


def typed_mask(lesions: dict) -> np.ndarray:
    """The same pixels, shaded by candidate type.

    Greyscale rather than colour on purpose: the mask is the thing a downstream
    tool would consume, and a grey level is a label while a colour is a rendering
    choice.
    """
    m = np.zeros((lesions["h"], lesions["w"]), np.uint8)
    flat = m.ravel()
    for c in lesions["candidates"]:
        flat[c["pixels"]] = C.LESION_TYPES[c["type"]]["grey"]
    return m


def lesion_overlay(rgb: np.ndarray, lesions: dict) -> np.ndarray:
    """Candidates marked on the fundus, one shape per type.

    Every marker shape is distinct. Labelling only the largest few left most marks
    anonymous, which made the overlay impossible to read.

    Only the largest are drawn once there are more than the overlay can carry -
    past that the marks overlap into a solid mass and the image stops carrying
    information. Both mask panels are unaffected and always show every pixel.
    """
    out = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR).copy()

    for c in lesions["candidates"][:C.OVERLAY_CIRCLE_LIMIT]:
        r = max(4, int(round(np.sqrt(c["area"] / np.pi) * 1.6)))
        x, y = int(round(c["cx"])), int(round(c["cy"]))
        col = _TYPE_COLOUR[c["type"]]

        if c["type"] in ("MA", "HEM"):
            cv2.circle(out, (x, y), r, col, 2, cv2.LINE_AA)
            if c["type"] == "HEM":
                cv2.circle(out, (x, y), int(r * 1.6), col, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(out, (x - r, y - r), (x + r, y + r), col, 2)

        # A candidate lying along the course of a vessel is kept - a microaneurysm
        # beside a venule is real and common - but struck through, because a wide
        # spot in a vessel looks exactly the same.
        if c["onVessel"]:
            cv2.line(out, (x - r, y + r), (x + r, y - r), (255, 255, 255), 1,
                     cv2.LINE_AA)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def anatomy_overlay(rgb: np.ndarray, anatomy: Optional[dict]) -> np.ndarray:
    """Landmarks with the quadrant axes centred on the optic disc."""
    out = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR).copy()
    if not anatomy:
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    h, w = out.shape[:2]
    d = anatomy["disc"]
    f = anatomy["fovea"]
    dx, dy = int(round(d["x"])), int(round(d["y"]))

    # Quadrant axes, which is what the ICDR rule counts hemorrhages against.
    for col, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.line(out, (0, dy), (w, dy), col, thick)
        cv2.line(out, (dx, 0), (dx, h), col, thick)

    cv2.circle(out, (dx, dy), int(round(d["radius"])), (255, 255, 255), 2,
               cv2.LINE_AA)
    cv2.circle(out, (int(round(f["x"])), int(round(f["y"]))),
               int(round(anatomy["maculaRadius"])), (255, 220, 120), 1,
               cv2.LINE_AA)

    s = max(4, int(w * 0.009))
    fx, fy = int(round(f["x"])), int(round(f["y"]))
    cv2.line(out, (fx - s, fy), (fx + s, fy), (255, 220, 120), 2, cv2.LINE_AA)
    cv2.line(out, (fx, fy - s), (fx, fy + s), (255, 220, 120), 2, cv2.LINE_AA)

    _label(out, "Optic disc", dx, dy - int(round(d["radius"])) - 6)
    _label(out, "Macula", fx, fy - int(round(anatomy["maculaRadius"])) - 6)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def cam_colour(rgb: np.ndarray, cam: dict) -> np.ndarray:
    """The activation map in colour tones, faded out at the analysed boundary."""
    base = np.asarray(rgb, np.float64)
    idx = np.clip(cam["camArray"] * 255, 0, 255).astype(np.uint8)
    heat = _LUT[idx].astype(np.float64)

    alpha = (0.45 * cam["feather"])[:, :, None]
    blend = base * (1 - alpha) + heat * alpha
    out = cv2.cvtColor(np.clip(blend, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    fld = cam["field"]
    cv2.circle(out, (int(round(fld["cx"])), int(round(fld["cy"]))),
               int(round(fld["r"])), (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def cam_grey(cam: dict) -> np.ndarray:
    """The same gradient in greyscale."""
    return np.clip(cam["camArray"] * 255, 0, 255).astype(np.uint8)


def cam_regions(rgb: np.ndarray, cam: dict, blobs: list) -> np.ndarray:
    """The attention regions, numbered - except the ones on the disc.

    Disc regions are drawn dashed-thin and labelled rather than numbered among the
    results. Attention there is expected, and presenting it as a finding would be
    misleading.
    """
    out = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR).copy()
    n = 0
    for b in blobs:
        x, y = int(round(b["cx"])), int(round(b["cy"]))
        r = max(9, int(round(b["radius"])))
        if b["onDisc"]:
            cv2.circle(out, (x, y), r, (200, 200, 200), 1, cv2.LINE_AA)
            _label(out, "disc", x, y - r - 6)
        else:
            n += 1
            cv2.circle(out, (x, y), r, (60, 220, 255), 2, cv2.LINE_AA)
            _label(out, str(n), x, y - r - 6)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _label(img, text, x, y):
    """Halo text, so a label stays readable on both dark and bright retina."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    org = (int(x - tw / 2), int(max(th + 2, y)))
    cv2.putText(img, text, org, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def to_png(img: np.ndarray) -> bytes:
    """Encode as PNG. Lossless, because these are masks as well as pictures."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", a)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
