"""Diabetic retinopathy screening pipeline.

A port of the algorithm in ``js/app.js``, stage for stage. Where the two could
reasonably differ - a binned statistic, a tile boundary - this implementation
reproduces the JavaScript's choice rather than the more obvious one, so that a
disagreement between them is a bug rather than a rounding difference. See
``python/README.md`` for the places that still differ and why.
"""

from .anatomy import (estimate_anatomy, estimate_fovea_macula,
                      estimate_optic_disc, quadrant_label)
from .enhance import enhance_image
from .grading import assess_severity, quadrant_lesion_counts
from .lesions import detect_lesion_candidates
from .pipeline import run_pipeline
from .quality import assess_quality, compute_metrics

__all__ = [
    "run_pipeline",
    "compute_metrics", "assess_quality", "enhance_image",
    "estimate_anatomy", "estimate_optic_disc", "estimate_fovea_macula",
    "quadrant_label",
    "detect_lesion_candidates",
    "assess_severity", "quadrant_lesion_counts",
]

__version__ = "1.0.0"
