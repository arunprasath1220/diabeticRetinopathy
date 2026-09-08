"""Every tuned constant in the DR screening pipeline, in one place.

The values are carried across unchanged from the reference implementation in
``js/app.js``. They are gathered here rather than scattered through the modules
because almost every one of them was arrived at by measurement, and a number you
cannot find is a number nobody can re-measure.

Where a constant encodes a decision rather than a scale, the reasoning is kept
with it. The reasoning is the part that is hard to reconstruct later.
"""

# ---------------------------------------------------------------------------
# Working sizes
# ---------------------------------------------------------------------------
WORKING_MAX_DIM = 800   # cap on the working/display image's long edge
METRIC_DIM = 512        # standardised size for quality metrics, so the
                        # thresholds below are resolution-independent

# ---------------------------------------------------------------------------
# Step 1: quality gate
# ---------------------------------------------------------------------------
QUALITY_THRESHOLDS = {
    "minFocusVar": 15,
    "enhanceFocusVar": 40,
    "minBrightMean": 40,
    "maxBrightMean": 235,
    "enhanceBrightLow": 60,
    "enhanceBrightHigh": 200,
    "minFov": 0.35,
    "enhanceFov": 0.5,
}

# ---------------------------------------------------------------------------
# Step 2/3: classifier and Grad-CAM
# ---------------------------------------------------------------------------
CLASS_NAMES = ["No DR", "DR present"]
MODEL_INPUT_SIZE = (224, 224)

# Grad-CAM layer. The last convolutional layer is only 7x7 at a 224px input, so
# upsampling it to the display gives roughly 114-pixel cells - the rectangular
# plateaus that make the heat map look blocky and put its edges nowhere near any
# actual lesion. The deepest 14x14 layer halves the cell size for the same kind
# of features.
CAM_GRID_HINT = 14
# How far above the retina's own typical activation a region must peak before it
# is marked. Grad-CAM is scaled to its own maximum, so *something* always reaches
# 1.0 - including on an image where the model found nothing. This is what
# separates a hotspot from the top of a flat map, and it is measured against the
# map's median absolute deviation over retina, so it follows each image rather
# than assuming a fixed contrast.
CAM_NOISE_K = 3.0
# A map whose retinal median is already this high is uniformly warm: there is no
# hotspot to speak of, only a ceiling, and circling its highest corner would
# invent a localisation the model never made.
CAM_FLAT_MEDIAN = 0.55
# The cosmetic fade at the edge of the analysed field, as a fraction of the trim
# itself. Drawing only - detection uses the hard mask.
FIELD_FEATHER_FRAC = 0.7
BLOB_THRESHOLD = 0.50
BLOB_MIN_AREA_FRAC = 0.0012
CAM_REGION_LIMIT = 12
OVERLAY_CIRCLE_LIMIT = 150

# ---------------------------------------------------------------------------
# Step 4: lesion detection
# ---------------------------------------------------------------------------
# Detection sensitivity. Thresholds are multiples of a robust noise estimate, so
# they follow each image rather than assuming one camera. Lowering DARK_K finds
# fainter lesions and more noise; MIN_CONTRAST_K is the backstop that keeps that
# trade from turning into speckle.
DARK_K = 3.5
BRIGHT_K = 4.0
MIN_CONTRAST_K = 3.0

# Absolute floors. Without them a very clean image drops its own threshold into
# the texture: the noise estimate falls to a couple of grey levels, and normal
# choroidal texture then reads as lesions.
DARK_FLOOR = 10
BRIGHT_FLOOR = 12
DARK_CONTRAST_FLOOR = 9
BRIGHT_CONTRAST_FLOOR = 11

# Cotton wool spots are a quarter to a half disc diameter across.
CWS_MIN_AREA_FRAC = 0.0004

# A large region far out toward the aperture rim is an artifact, not a lesion:
# shadows from pupil misalignment, peripapillary vignetting. Genuine peripheral
# lesions are small, so the rule is keyed on size as well as position.
EDGE_ZONE_FRAC = 0.80
EDGE_ARTIFACT_AREA_FRAC = 0.0015

# Permissive threshold as a fraction of the strict one. Recovers cluster members
# and faint lesion margins that a single threshold cuts off.
HYST_RATIO = 0.45

# Vessel map. A vessel is told from a lesion by the *spread* of the directional
# closing across orientations, not by the size of any one response: at a vessel
# the element lying along it fills nothing while every crossing element fills it
# completely, so the responses are far apart. At a round lesion they are close.
# Comparing the two responses to each other rather than to a fixed number is
# scale-free, so one rule covers a thin peripheral capillary and a wide vein at
# the disc alike.
VESSEL_ELONGATION = 1.5
VESSEL_SEED_Q = 0.95
VESSEL_GROW_Q = 0.80
# A region is only discarded as vessel when this much of it lies on the vessel
# map. Set too low, lesions touching a vessel are lost with it.
VESSEL_OVERLAP_REJECT = 0.78

# Half-length of the linear structuring element that separates lesions from
# vessels, as a fraction of the short edge. Too short and thin vessels are
# filled and read as lesions; too long and short vessel segments read as lesions.
LINEAR_SE_FRAC = 0.014
LINEAR_ORIENTATIONS = 12
SEED_TAIL_Q = 0.97
SEED_FLOOR_DARK = 6
SEED_FLOOR_BRIGHT = 8

# Scale saturation. A bounded blob is already filled at the element matching its
# own width, so the next element up adds nothing; anything that continues past
# the element keeps gaining at every rung. That, not amplitude, is what separates
# a lesion from a vessel bend or a patch of choroidal mottling.
SATURATION_SE_MULT = 2.0
SATURATION_LADDER = 3           # seL, 2*seL, 4*seL
SATURATION_ORIENTATIONS = 6
SATURATION_MAX_GROWTH = 1.10

# Vessel junction test. Where two vessels cross, where one branches and at the
# apex of a tight bend there is no direction in which the structure is straight,
# so every orientation fills it and it answers the roundness test exactly as a
# lesion does. What differs is what runs out of it.
ARM_DIRECTIONS = 24
ARM_PERSISTENCE = 0.55
ARM_COLLAR_PAD = 3
JUNCTION_ANTIPODAL_TOL_DEG = 45
JUNCTION_RING_SATURATION = 0.70
ON_VESSEL_CONTRAST_FACTOR = 1.35

# Broad, smooth darkening is anatomy or shadow, not a lesion.
SMOOTH_GRADIENT_RATIO = 0.55
MACULA_SMOOTH_AREA_FRAC = 0.08
SMOOTH_BROAD_AREA_FRAC = 0.004

# Above this many candidates the detector is almost certainly responding to
# image noise, and the report says so instead of presenting a tidy count.
NOISE_SUSPICION_COUNT = 400

# ---------------------------------------------------------------------------
# Peripapillary bright suppression
# ---------------------------------------------------------------------------
# The optic disc is the brightest thing in a normal fundus, so every bright
# candidate it produces is a false exudate. Excluding a circle of 1.15 disc radii
# only works as well as the radius does, and a radius read off a plateau at a
# fixed height above the *global* retinal mean lands inside the rim. The rim, a
# scleral crescent and the peripapillary reflex then fall in the gap between
# where the circle stops and where the disc ends - and each is bright, round and
# sharply bounded, which is the description of a hard exudate.
PERIPAPILLARY_MULT = 2.5
DISC_EDGE_LEVEL = 0.50      # half-maximum: where a blurred edge actually sits
DISC_HALO_LEVEL = 0.28      # lower, to take in crescent and reflex
DISC_FILL_MAX_MULT = 3.2    # past this the fill has escaped into open retina
DISC_TISSUE_OVERLAP = 0.33  # of a bright region, before the disc explains it
DISC_COLOUR_LEVEL = 0.55    # on the retina(0)->disc(1) blue-fraction scale
DISC_COLOUR_MIN_SEPARATION = 0.02   # below this the reference is degenerate

# ---------------------------------------------------------------------------
# Step 5: ICDR grading
# ---------------------------------------------------------------------------
# The "4" arm of the 4-2-1 rule for severe NPDR: more than twenty intraretinal
# hemorrhages in each of the four quadrants.
ICDR_HEM_PER_QUADRANT = 20
QUADRANT_NAMES = ["Superior-Nasal", "Superior-Temporal",
                  "Inferior-Nasal", "Inferior-Temporal"]

# ---------------------------------------------------------------------------
# Lesion classes
# ---------------------------------------------------------------------------
LESION_ORDER = ["MA", "HEM", "HE", "CWS"]
LESION_TYPES = {
    "MA":  {"key": "MA",  "label": "Microaneurysm candidate",
            "grey": 255, "shape": "circle",     "shapeName": "circle"},
    "HEM": {"key": "HEM", "label": "Dot/blot hemorrhage candidate",
            "grey": 200, "shape": "ring2",      "shapeName": "double circle"},
    "HE":  {"key": "HE",  "label": "Hard exudate candidate",
            "grey": 150, "shape": "square",     "shapeName": "square"},
    "CWS": {"key": "CWS", "label": "Cotton wool spot candidate",
            "grey": 100, "shape": "squareDash", "shapeName": "dashed square"},
}
