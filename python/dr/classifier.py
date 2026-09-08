"""Step 2 and Step 3a: the classifier and its explanation.

Kept behind a single small class so the rest of the pipeline never imports torch.
That is what lets a deployment with no model file, or no deep-learning stack at
all, still produce a lesion map and an ICDR estimate - see :mod:`dr.pipeline`.

On matching the browser demo
----------------------------
The JavaScript loads a specific MobileNet through TensorFlow.js. This loads a
MobileNetV2 trained by :func:`dr.train.train_classifier`. Different weights mean
Steps 2 and 3a will not agree numerically between the two, and are not expected
to. Steps 1, 3, 4 and 5 are deterministic image processing and do agree - see the
parity tool in ``tools/parity``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config as C
from .field import aperture_geometry, retina_field_mask
from .imaging import box_blur, erode_mask, image_channels
from .stats import cam_field_stats


class Classifier:
    """A binary DR classifier plus Grad-CAM, loaded from a checkpoint."""

    def __init__(self, model, device: str = "cpu",
                 feature_layer: Optional[str] = None):
        self.model = model
        self.device = device
        self.feature_layer = feature_layer
        self.model.eval()

    # -- construction ----------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str] = None, device: str = "cpu"):
        """Load a checkpoint, or return ``None`` if there is not one.

        Returning ``None`` rather than an untrained backbone is deliberate. A
        MobileNet with a randomly initialised head will happily emit a
        confident-looking probability that means nothing at all, and on a
        screening report that is worse than no number.
        """
        import torch

        if path is None:
            path = Path(__file__).resolve().parent.parent / "models" / "dr_classifier.pt"
        path = Path(path)
        if not path.is_file():
            return None

        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = build_model(num_classes=len(C.CLASS_NAMES))
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        model.to(device)
        return cls(model, device=device,
                   feature_layer=(ckpt.get("feature_layer")
                                  if isinstance(ckpt, dict) else None))

    # -- preprocessing ---------------------------------------------------
    def _prepare(self, rgb: np.ndarray):
        """Put an image into the form the network was trained on.

        Written once and used by both :meth:`classify` and :meth:`grad_cam`. If
        the two ever disagreed, the heatmap would be explaining a slightly
        different image from the one that was graded - an explanation that is
        wrong in a way nobody would notice by looking at it.
        """
        import torch

        resized = cv2.resize(np.asarray(rgb), C.MODEL_INPUT_SIZE,
                             interpolation=cv2.INTER_NEAREST)
        x = resized.astype(np.float32)
        x = (x - 127.5) / 127.5
        x = np.transpose(x, (2, 0, 1))[None]
        return torch.from_numpy(x).to(self.device)

    # -- Step 2 -----------------------------------------------------------
    def classify(self, rgb: np.ndarray) -> dict:
        import torch

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(self._prepare(rgb))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        return {"probs": [float(p) for p in probs],
                "predClass": int(np.argmax(probs)),
                "elapsedMs": (time.perf_counter() - t0) * 1000}

    # -- Step 3a ----------------------------------------------------------
    def grad_cam(self, rgb: np.ndarray, class_index: int = 1,
                 anatomy: Optional[dict] = None) -> dict:
        """Where the classifier looked, confined to the retina.

        Always computed for the "DR present" class regardless of what the network
        predicted. An explanation of why the model said "no disease" is not what a
        clinician needs; what they need is where the evidence for disease would
        have been.
        """
        import torch

        target = self._feature_module()
        activations = {}
        gradients = {}

        def fwd_hook(_m, _i, out):
            activations["v"] = out.detach()

        def bwd_hook(_m, _gi, gout):
            gradients["v"] = gout[0].detach()

        h1 = target.register_forward_hook(fwd_hook)
        h2 = target.register_full_backward_hook(bwd_hook)
        try:
            x = self._prepare(rgb)
            logits = self.model(x)
            self.model.zero_grad(set_to_none=True)
            logits[0, class_index].backward()
        finally:
            h1.remove()
            h2.remove()

        act = activations["v"][0]          # (C, h, w)
        grad = gradients["v"][0]
        weights = grad.mean(dim=(1, 2), keepdim=True)
        raw = torch.relu((weights * act).sum(dim=0)).cpu().numpy()

        h, w = np.asarray(rgb).shape[:2]
        raw = cv2.resize(raw.astype(np.float32), (w, h),
                         interpolation=cv2.INTER_LINEAR)
        raw = np.maximum(raw, 0)

        cam = confine_cam_to_retina(rgb, raw)
        cam["description"] = describe_grad_cam(cam, anatomy)
        cam["classIndex"] = class_index
        return cam

    def _feature_module(self):
        """The deepest convolutional stage worth explaining from.

        The very last one gives the sharpest class discrimination but the coarsest
        spatial grid - 7x7 at a 224px input, about 32 pixels per cell, which is
        wider than many of the lesions being explained. One stage earlier doubles
        the resolution for a modest loss of specificity, which is the better trade
        when the output is a localisation claim.
        """
        if self.feature_layer:
            return dict(self.model.named_modules())[self.feature_layer]
        features = getattr(self.model, "features", None)
        if features is None:
            raise ValueError("no usable convolutional layer for Grad-CAM")
        # The 14x14 stage of MobileNetV2 at a 224px input.
        idx = min(13, len(features) - 1)
        return features[idx]


def build_model(num_classes: int = 2):
    """MobileNetV2 with a fresh head.

    A fundus photograph shares low-level structure with natural images - edges,
    blobs, texture - but nothing at all above that, so freezing the backbone
    entirely underfits and training from scratch on a screening-sized dataset
    overfits. Fine-tuning the whole thing at a low rate with a fast head is the
    middle path; see :mod:`dr.train`.
    """
    import torch.nn as nn
    from torchvision.models import mobilenet_v2

    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def confine_cam_to_retina(rgb: np.ndarray, raw: np.ndarray) -> dict:
    """Mask, normalise and characterise a raw activation map.

    A Grad-CAM cell that straddles the aperture edge is computed from a receptive
    field that is mostly black surround, so its value says nothing about retina -
    but it is often large, and it lands in a ring around the image where it looks
    like a finding. The map is therefore masked to the retina eroded by half a CAM
    cell, and the boundary is returned so the UI can draw it, turning an otherwise
    unexplained ring into a stated exclusion.

    The hard mask and the soft one are deliberately different things. Painting an
    overlay through the hard mask leaves a step at its edge - analysed retina
    tinted, the trimmed rim not - and that step reads as a rendering fault rather
    than as the edge of the analysed field. The feathered copy fades instead.
    Detection still uses the hard mask: what is measured and what is painted are
    different questions.
    """
    ch = image_channels(rgb)
    w, h, lum = ch["w"], ch["h"], ch["lum"]

    ap = aperture_geometry(lum)
    inside = retina_field_mask(lum, 0.05)

    bleed = max(2, int(round(min(w, h) / (C.CAM_GRID_HINT * 2))))
    inside = erode_mask(inside, bleed)

    feather = box_blur(inside.astype(np.float64),
                       int(np.clip(round(bleed * C.FIELD_FEATHER_FRAC), 3, 24)))

    field = {"cx": ap["cx"], "cy": ap["cy"],
             "r": max(1.0, ap["R"] * 0.955 - bleed)}

    inside_count = int(inside.sum())
    raw_max = float(raw[inside].max()) if inside_count else 0.0

    cam_array = np.zeros((h, w))
    if inside_count == 0 or raw_max <= 1e-12:
        return {"camArray": cam_array, "inside": inside, "feather": feather,
                "field": field, "insideCount": inside_count, "rawMax": raw_max,
                "h": h, "w": w, "stats": {"median": 0.0, "mad": 0.0}}

    cam_array[inside] = raw[inside] / raw_max
    return {"camArray": cam_array, "inside": inside, "feather": feather,
            "field": field, "insideCount": inside_count, "rawMax": raw_max,
            "h": h, "w": w, "stats": cam_field_stats(cam_array, inside)}


def describe_grad_cam(cam: dict, anatomy: Optional[dict]) -> dict:
    """Say in words what the attention map looks like.

    ``concentration`` is the share of total activation held by the warmest tenth
    of the retina. It is what separates a map with one hotspot from a map that is
    uniformly warm - and the peak value cannot make that distinction, because
    normalisation guarantees a peak of 1.0 either way.

    Every statistic is over retina only. Measured over the whole frame the black
    surround contributes a large block of zeros that inflates the concentration
    figure, and the peak it reports can land outside the eye entirely.
    """
    from .anatomy import quadrant_label

    cam_array = cam["camArray"]
    inside = cam["inside"]
    v = cam_array[inside]

    if v.size == 0:
        return {"region": "n/a", "focal": False, "flat": True,
                "concentration": 0.0, "median": 0.0, "peakX": 0, "peakY": 0}

    masked = np.where(inside, cam_array, -np.inf)
    py, px = np.unravel_index(int(np.argmax(masked)), cam_array.shape)

    if anatomy:
        region = quadrant_label(px, py, anatomy["disc"], anatomy["nasalSide"])
    else:
        region = ("superior frame" if py < cam_array.shape[0] * 0.5
                  else "inferior frame")

    ordered = np.sort(v)[::-1]
    top_n = max(1, int(round(ordered.size * 0.1)))
    total = float(v.sum())
    concentration = float(ordered[:top_n].sum() / total) if total > 1e-8 else 0.0

    return {"region": region,
            "focal": concentration > 0.35,
            "flat": cam["stats"]["median"] >= C.CAM_FLAT_MEDIAN,
            "concentration": concentration,
            "median": cam["stats"]["median"],
            "peakX": int(px), "peakY": int(py)}


def find_defect_blobs(cam: dict, anatomy: Optional[dict]) -> list:
    """Discrete regions of concentrated model attention.

    Two things decide the threshold, and the second is why an image with nothing
    to find now produces no circles at all. Grad-CAM is scaled to its own maximum,
    so some pixel is always 1.0 and a fixed fraction-of-peak cut always returns a
    region - on a healthy retina, the tallest bump in a flat map. Requiring the
    region also to stand clear of the retina's own typical activation removes
    exactly that case, and leaves a genuine hotspot untouched.
    """
    from scipy import ndimage as ndi

    from .anatomy import quadrant_label

    cam_array = cam["camArray"]
    inside = cam["inside"]

    if cam["stats"]["median"] >= C.CAM_FLAT_MEDIAN:
        return []                       # warm everywhere: no localisation

    noise_cut = cam["stats"]["median"] + C.CAM_NOISE_K * cam["stats"]["mad"]
    cut = max(C.BLOB_THRESHOLD, noise_cut)

    bw = inside & (cam_array >= cut)
    min_area = max(20, int(round(cam_array.size * C.BLOB_MIN_AREA_FRAC)))

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    lab, n = ndi.label(bw, structure=structure)

    blobs = []
    for k in range(1, n + 1):
        sel = lab == k
        area = int(sel.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(sel)
        cx, cy = float(xs.mean()), float(ys.mean())
        blob = {"cx": cx, "cy": cy, "area": area,
                "peak": float(cam_array[sel].max()),
                # Area-equivalent radius, widened a little so the marker encloses
                # rather than bisects the region it is drawn around.
                "radius": float(np.sqrt(area / np.pi) * 1.25)}
        # The optic disc draws strong attention from almost any fundus
        # classifier: it is the brightest, most distinctive structure in the frame
        # and the network uses it to orient itself. That attention is real and
        # worth showing, but it is not a finding.
        if anatomy:
            blob["onDisc"] = bool(np.hypot(cx - anatomy["disc"]["x"],
                                           cy - anatomy["disc"]["y"])
                                  <= anatomy["disc"]["radius"] * 1.2)
            blob["quadrant"] = quadrant_label(cx, cy, anatomy["disc"],
                                              anatomy["nasalSide"])
        else:
            blob["onDisc"] = False
            blob["quadrant"] = "n/a"
        blobs.append(blob)

    blobs.sort(key=lambda b: (-b["peak"], -b["area"]))
    return blobs[:C.CAM_REGION_LIMIT]
