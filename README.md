# Diabetic Retinopathy Screening — SIH26038

A browser-based screening, explainability and capacity-planning pipeline for retinal fundus
photographs. Built for Smart India Hackathon problem statement SIH26038 (sponsor: MathWorks).

Everything runs client-side. No image is uploaded anywhere, and there is no backend.

---

## Contents

1. [Quick start](#quick-start)
2. [The problem](#the-problem)
3. [Project layout](#project-layout)
4. [How it works, stage by stage](#how-it-works-stage-by-stage)
5. [The lesion detector in detail](#the-lesion-detector-in-detail)
6. [Anatomical landmarks](#anatomical-landmarks)
7. [What is real and what is not](#what-is-real-and-what-is-not)
8. [Tuning](#tuning)
9. [Testing](#testing)
10. [Bugs found along the way](#bugs-found-along-the-way)
11. [Known limitations](#known-limitations)
12. [Replacing the detector with a trained model](#replacing-the-detector-with-a-trained-model)
13. [Credits and licence](#credits-and-licence)

---

## Quick start

Open `index.html` in a browser. That is the whole installation.

- **`index.html`** — overview of the project, with diagrams.
- **`screening.html`** — the working application. Upload a fundus image and run the pipeline.
- **`reference.html`** — ICDR severity scale, lesion vocabulary, and a summary of the method.

The classifier weights (about 13 MB) download from a public CDN the first time
`screening.html` loads, so that page needs an internet connection once. If the download fails the
app says so plainly and refuses to show a grade, rather than substituting placeholder numbers.

Opening the files directly from disk works. There is no build step and no package manager.

---

## The problem

India has roughly 77 million diabetic adults. About 18% of them develop diabetic retinopathy, a
leading cause of preventable blindness. Early screening prevents around 90% of the resulting vision
loss.

The obstacle is not medicine, it is arithmetic. Rural India has roughly one ophthalmologist per
100,000 people. Manual screening cannot reach the population that needs it.

Automated tools exist, but those deployed so far tend to be opaque, thinly validated, and brittle on
the inconsistent images that low-cost portable cameras produce. This project aims at the opposite:
every stage shows its working, and every stage states plainly what it does not know.

---

## Project layout

```
index.html        Overview and diagrams
screening.html    The application: upload, pipeline, results, printable report
reference.html    Clinical reference and method summary
css/style.css     All styling for every page
js/app.js         The entire engine
README.md         This file
```

`js/app.js` is one IIFE with no dependencies other than TensorFlow.js, which `screening.html`
loads from a CDN. It is deliberately not an ES module, because ES modules are blocked when a page is
opened over `file://` and that would break the "just open it" property.

The file is organised in the order the pipeline runs: configuration, small helpers, model loading,
image handling, quality metrics, enhancement, inference, Grad-CAM, landmark estimation, lesion
detection, rendering, throughput simulation, report, and finally the event wiring.

---

## How it works, stage by stage

An image passes through five stages. **The two stages that point at disease do so by completely
unrelated means, and are never merged into a single score.** When they disagree, that disagreement
is information; averaging it away would hide it.

### Stage 1 — Image quality

Three real measurements, all from the pixels:

| Measure | Method |
| --- | --- |
| Focus | Variance of a Laplacian-filtered greyscale image |
| Brightness | Mean and standard deviation of the greyscale histogram |
| Field of view | Fraction of the frame that is not black border |

Metrics are computed at a standardised 512px working size so the thresholds mean the same thing
regardless of the uploaded image's resolution.

An image that fails outright is **rejected with the specific reason** ("mean brightness too low —
recapture with more illumination") and the pipeline stops. It is not screened anyway.

A borderline image is enhanced first: light denoising, illumination flattening by subtracting a
large-radius local mean, then CLAHE-style local contrast equalisation over an 8×8 tile grid with a
clipped and redistributed histogram and bilinear interpolation between tiles. Colour is preserved by
working in a luminance/chrominance space and only modifying luminance.

The thresholds are prototype heuristics, not clinically validated cutoffs, and the app says so.

### Stage 2 — Classification

A MobileNet trained on APTOS 2019 fundus images, converted to TensorFlow.js, running entirely in the
browser. Preprocessing matches what the model was trained with exactly: nearest-neighbour resize to
224×224, then scaling to the range −1 to 1. Mismatched preprocessing silently destroys accuracy, so
this is deliberately faithful to the original rather than "improved".

**This model has two classes, not five.** It reports no disease against disease present. It is not
an ICDR severity grader, and the app never presents one. A validated five-class model could not be
found in a form loadable in a browser, and deriving five levels from a two-class output would be a
fabrication.

**Confidence figures are raw softmax outputs, not calibrated probabilities.** A reading of 100% means
the image sits far on one side of the model's decision boundary. It does not mean the diagnosis is
100% likely to be correct, and a confidently wrong answer looks exactly the same. The app states this
next to every confidence figure, because the number invites the opposite reading.

### Stage 3 — Explainability (Grad-CAM)

Grad-CAM computed from real gradients of the model's output with respect to an internal
convolutional layer, following the standard formulation: gradients are averaged per channel to give
weights, the activation maps are combined with those weights, and the result is rectified and
normalised.

It is always computed against the "disease present" class, so it shows what evidence the model finds
for disease even when its final answer is "no disease".

Two implementation details matter:

- **Layer choice.** The last convolutional layer is only 7×7 at this model's input size, which
  becomes roughly 114-pixel blocks when scaled to the display and makes the map look rectangular. The
  app uses `conv_pw_11_relu`, the deepest 14×14 layer, halving the cell size. Everything after that
  layer is replayed in sequence to get the gradient, which is valid because MobileNet has no skip
  connections.
- **The optic disc.** Almost any fundus classifier attends strongly to the disc: it is the most
  distinctive structure in the frame and the network uses it to orient itself. That attention is
  real, so it is shown, but disc regions are drawn dashed and labelled rather than counted as
  findings.

Grad-CAM is presented in both a colour palette and greyscale. Both show identical values. The colour
version is the conventional presentation and easier to read at a glance; the greyscale one prints
reliably. The colour bar legend is generated from the same stops as the renderer, and a test asserts
they match, because a legend that disagrees with the pixels is worse than no legend.

**Grad-CAM answers "what drove this prediction", not "where are the lesions".** Those are different
questions with different answers.

### Stage 4 — Lesion candidates

Classical image processing. No machine learning at all, and completely independent of stage 2: it
did not influence the classification and the classification did not influence it. See
[the next section](#the-lesion-detector-in-detail) for the algorithm.

Output is three views: a binary mask of candidate pixels, the same mask shaded by candidate type, and
the candidates ringed on the fundus image. Each lesion type gets its own marker shape so every mark
is identifiable without reading a label.

**Every region is a candidate, not a finding.** Nothing here has been validated against ground-truth
lesion masks, so no accuracy figure is claimed.

### Stage 5 — Throughput planning

Explicitly not part of the AI. A capacity model that takes image acquisition rate, available
bandwidth, average file size, model throughput and ophthalmologist review capacity, and reports the
daily capacity of each stage, which one is the bottleneck, whether a backlog accumulates, and how
long a target population would take to screen.

Model throughput is **measured live** from the actual inference time on your device rather than
assumed.

A production version of this analysis would be a Simulink model in MATLAB. This is a lightweight
stand-in for early planning.

---

## The lesion detector in detail

This is the part that took the most work, and the obvious approach is wrong.

### Why size cannot separate lesions from vessels

The instinct is to filter by size: remove anything too thin or too small and the blood vessels go
away. **This cannot work, because a microaneurysm is smaller than a vessel is wide.** Any filter
narrow enough to remove vessels removes every small lesion with them.

An earlier version of this project did exactly that, using a morphological closing with a square
21-pixel element. It classified every microaneurysm and most dot hemorrhages as "vessel" and
discarded them. That was the single worst bug in the project.

### What actually separates them: elongation

A vessel keeps going. A lesion does not.

The image is probed with **linear structuring elements at twelve orientations**. A closing along a
line fills any dark structure shorter than the line in that direction:

- A **vessel** is longer than the element along the direction it runs, so it survives that
  orientation's closing.
- A **round lesion** is shorter than the element in every direction, so every orientation fills it.

Taking the minimum closing across orientations therefore leaves vessels dark and fills lesions.
Subtracting the original gives a map that responds to round dark structures and ignores vessels. The
same construction with openings and a maximum does the equivalent for bright structures, which is
what separates an exudate from the bright reflex running along an arteriole.

Twelve orientations is not arbitrary. Four (the axes and diagonals) leaves vessels at intermediate
angles unprotected: no element lies along them, every orientation cuts across, and they get filled
exactly like lesions. Eight was measured to be worse than twelve on the test fixture, so twelve is
what ships.

**This also excludes the macula for free.** The response is proportional to how *steeply* intensity
changes across the element, so a lesion's step edge answers strongly while the macula's gradual
darkening barely answers at all, however much darker the macula is overall. No rule about where the
macula is was needed.

### Seeds and growth

The directional maps decide *where* lesions are. Their *extent* comes from a separate multi-scale
background subtraction, joined by hysteresis: a region must contain a pixel clearing a strict
threshold before it exists at all, and is then grown outward to a much lower one.

This matters because a single threshold forces one choice for the whole image. Raise it to remove
false marks and faint real lesions go too. Hysteresis breaks that trade: background texture never
produces a core strong enough to start a region, so it is never grown, while a faint cluster member
connected to a strong core is recovered.

Background subtraction runs at **three scales**. A lesion larger than the averaging window
contaminates the background it is measured against, so its contrast collapses and it vanishes. One
scale misses large lesions; two still miss confluent clusters, where neighbouring lesions raise each
other's local background and the whole group fades out together. The coarsest scale measures the
group against general retina instead.

Thresholds are multiples of a **median absolute deviation** noise estimate rather than a standard
deviation, because the vessel tree is a large population of strong dark deviations that inflates the
standard deviation and pushes the threshold above real lesions.

### Guards against specific false positives

| Guard | Why it exists |
| --- | --- |
| Background excludes vessels | A vessel inside the averaging window drags the local background down, so ordinary retina *between* two vessels reads as abnormally bright and appears as a soft-edged bright finding along the arcades. |
| Background excludes the black surround | A plain neighbourhood average near the field-of-view rim mixes in the black border, making ordinary edge retina read as abnormally bright. Uncorrected this rings every image in spurious findings. |
| Growth stops at vessels | A lesion lying on a vessel seeds correctly, but the permissive mask covers the vessel too, so the region runs out along the whole vessel tree and is then discarded as oversized, taking the lesion with it. |
| Nothing seeded is ever dropped | If a region fails a size or shape test, the seed inside it is reconsidered on its own extent. The evidence for a lesion is the seed, not how far the region grew. |
| Large regions at the aperture rim | Pupil-misalignment shadows, peripapillary crescents and flare all sit at the edge of the field and can be as large and dark as a hemorrhage. Keyed on size as well as position, so small peripheral lesions survive. |
| Optic disc exclusion | The disc is the brightest structure in a normal retina, and the vessels converging on it plus its dark cup would otherwise produce both bright and dark false candidates. |

### Type assignment

| Type | Rule |
| --- | --- |
| Microaneurysm | Small, round, isolated, darker than local background |
| Dot/blot hemorrhage | Same darkness test, larger area, still compact |
| Hard exudate | Brighter than local background with a sharp edge, measured as mean gradient magnitude |
| Cotton wool spot | Brighter than background, soft low-gradient edge, at least a quarter disc diameter across |

**The cotton wool spot class is the least trustworthy.** A soft edge is also what a defocused
exudate, a haze artefact or a smudge on the lens looks like, and edge sharpness alone cannot separate
them. Treat anything marked this way as unidentified bright material.

The microaneurysm/hemorrhage split rests on area alone and will misassign borderline sizes.

---

## Anatomical landmarks

The app estimates the optic disc, fovea and macula, and uses them to name retinal quadrants.

**Optic disc.** Scored on brightness *and* local contrast together, because the disc carries the
vessel trunk and a sharp rim and is therefore bright and textured, while lens flare is bright and
smooth. Brightness alone hands the disc to any highlight in the frame. The centre is the centroid of
the bright plateau rather than a single peak pixel, and the radius is measured from that plateau
instead of assumed.

**Fovea.** Measured, not merely placed. The vessels are first removed by a morphological closing,
which deletes dark structures thinner than the filter rather than merely diluting them as a blur
does; without this the darkest region found is usually a vessel bundle. The remaining broad darkening
is then scored against an anatomical anchor, since the fovea sits about 2.5 disc diameters temporal
to the disc and slightly below its level. Darkness alone let any dark structure in the search window
capture the marker. The anchor holds the estimate where the fovea has to be while genuine macular
darkening still moves it, and with no dark signal at all it degrades gracefully to the anatomical
position rather than snapping to noise.

**Nasal and temporal** follow from the measured disc-to-fovea vector. The optic disc lies nasal to
the fovea in *both* eyes, so the side the disc sits on is the nasal side, and no left/right eye
information is needed. Superior is taken as the top of the frame, per fundus photography convention.

These are heuristics, not a trained landmark detector, and nothing here is validated against
annotated landmark positions. A large hemorrhage inside the search zone can be darker than the true
fovea and capture the marker. A mislocated disc puts the search in the wrong place and every
downstream quadrant name inherits the error.

---

## What is real and what is not

| Output | What it really is |
| --- | --- |
| Image quality scores | Real measurements. Thresholds are heuristics, not validated cutoffs. |
| Classification | Genuine inference from a real model. **Two classes only.** |
| Confidence figures | Raw softmax. **Not calibrated probabilities.** |
| Grad-CAM | Real gradients. Coarse; a boundary means "around here". |
| Lesion candidates | Classical morphology, independent of the classifier. **Candidates, not findings.** |
| Severity grade | **Not produced.** No validated five-class model was loadable. |
| Sensitivity / specificity | **Not claimed.** No peer-reviewed validation exists for this model. |
| Segmentation | **Not learned segmentation.** The morphological detector is not a substitute. |
| Throughput simulation | Arithmetic on your inputs, with model speed measured live. |

This is a research and hackathon prototype. It is not a certified medical device and nothing it
produces is a diagnosis.

---

## Tuning

Detection sensitivity depends on your camera and your images, and no single default suits every set.
The app exposes the thresholds directly rather than hiding them.

**In Step 4 (lesion detection):**

| Control | Effect |
| --- | --- |
| Dark-lesion sensitivity | How strong a response must be before a dark region counts. Lower finds fainter microaneurysms and hemorrhages, and more noise with them. |
| Bright-lesion sensitivity | The same, for exudates. |
| Minimum contrast above noise | How far a region must stand clear of the image's own noise. **Raise this first if you get marks on plain retina**, before touching the other two. |

**In Step 3 (Grad-CAM):** attention threshold and maximum region count. Both re-draw without
re-running the model.

Both panels re-run only their own stage, so iterating takes seconds.

In `js/app.js`, the constants block near the top holds the rest: `LINEAR_SE_FRAC` and
`LINEAR_ORIENTATIONS` for the directional morphology, `HYST_RATIO` for how far regions grow from
their cores, `VESSEL_SE_FRAC` for the assumed maximum vessel width, and the area fractions that
bound candidate sizes. Each is commented with what it trades off.

---

## Testing

The algorithms are pure computation, so they are tested directly in Node against functions extracted
from `js/app.js`. No browser is needed.

| Suite | Coverage |
| --- | --- |
| Seed level | 20 checks: lesions from microaneurysm to blot size, a lesion on a vessel, an exudate cluster, vessels at five angles, macula samples, plain retina |
| End to end | 34 checks: the full detector, including a hemorrhage on each of four differently angled vessels, a vignetted rim sampled all the way round, a crescent shadow, a large edge artifact, and a small peripheral lesion that must survive |

The end-to-end suite exists because a seed-only suite let a real bug through: seeding was correct,
but the region then grew along the vessel and the whole component was discarded on size. **Testing a
stage in isolation is not the same as testing the pipeline.**

The fixtures carry self-checks, because three separate test failures turned out to be faults in the
fixture rather than the code:

- Vessels must be rasterised by perpendicular distance. Stepping along a centreline with rounding
  leaves pinholes, and one background pixel inside the window makes the closing fill the vessel,
  which looks exactly like an algorithm bug.
- Every planted lesion must have measurable contrast against its own surroundings. A lesion drawn at
  the same value as a shadow it sits on is invisible by construction.
- No probe that must report nothing may sit on a planted feature.

---

## Bugs found along the way

Worth recording, because several were invisible from the output and only showed up under
measurement.

1. **Square structuring element classified lesions as vessels.** A closing with a square 21-pixel
   element removes everything smaller than itself, and a microaneurysm is smaller than a vessel is
   wide. Every small lesion was being discarded as vessel. Fixed by switching to directional
   elements.
2. **Four orientations left oblique vessels unprotected.** Vessels at angles between the axes and
   diagonals were filled and reported as lesions. Caught by a test with vessels at five angles.
3. **Half-resolution morphology wrecked vessel discrimination.** A speed optimisation. Measured at
   579 false vessel seeds against 25 at full resolution, with identical sensitivity. Reverted.
4. **Dark lesions punched holes in their own analysis region.** The retinal field was decided by one
   brightness threshold, and a hemorrhage is dark enough to fall below it, so it was carved out as
   though it lay outside the camera's view. Fixed with separate thresholds for fitting the aperture
   and for deciding membership.
5. **The field mask trimmed the outer tenth of the retina**, discarding genuine peripheral lesions.
6. **Growth ran along vessels.** A lesion on a vessel seeded correctly, then grew out along the whole
   vessel tree and was discarded as oversized, taking the lesion with it.
7. **Unmasked background averaging at the field rim** made ordinary edge retina read as abnormally
   bright, producing a ring of spurious bright findings.
8. **Blur and noise estimation called a helper several million times per pass.** Inlining took the
   blur stage from 7077 ms to 119 ms and the noise estimate from 2784 ms to 48 ms.

---

## Known limitations

- **Flame-shaped hemorrhages and neovascularisation are largely invisible** to the lesion detector.
  Both are elongated, so the vessel test discards them. The findings that define proliferative
  disease are exactly the ones it misses.
- **Venous beading and intraretinal microvascular abnormalities are not detected at all.** Both
  require judging vessel calibre along its length.
- **No severity grading.** The classifier has two classes.
- **Grad-CAM is coarse.** Even at 14×14 the cells are large relative to a microaneurysm.
- **Detection takes several seconds** and runs on the main thread. A status line paints before the
  slow stage, but the page is unresponsive while it works. A Web Worker would fix this.
- **Landmark estimates can be wrong** on images with bright artefacts, heavy vignetting, a disc
  cropped at the frame edge, or a photograph not centred on the posterior pole.
- **Nothing is validated against clinical ground truth.**

---

## Replacing the detector with a trained model

A learned segmentation model would outperform the morphological detector, and a suitable one already
exists rather than needing to be trained from scratch:

**[ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d](https://huggingface.co/ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d)**
— a U-Net with an SE-ResNeXt50 encoder, trained on DDR, FGADR, IDRiD, MESSIDOR and RETLES, MIT
licensed.

It cannot be loaded as things stand. It ships as PyTorch `safetensors`, which a browser cannot run,
and no TensorFlow.js or ONNX export is published. Getting it into this page takes three steps on a
machine with Python and PyTorch:

1. Export the checkpoint to ONNX.
2. Either serve it with ONNX Runtime Web, or convert it onward to TensorFlow.js with
   `tensorflowjs_converter`.
3. Host the result at a URL that permits cross-origin requests, as the Step 2 classifier already is.

**This is a packaging problem, not a research one.** It is the highest-value work left in the
project, and it would let most of the morphology be deleted rather than tuned further.

---

## Credits and licence

**Classifier:** MobileNet-based binary diabetic retinopathy detector converted to TensorFlow.js by
[vbookshelf/Diabetic-Retinopathy-Analyzer](https://github.com/vbookshelf/Diabetic-Retinopathy-Analyzer),
trained via the Kaggle kernel *DR - MobileNet Binary Classifier + TFJS Web App* on the
[APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset.
Weights load at runtime from that public repository.

**Grading scale:** International Clinical Diabetic Retinopathy (ICDR) severity scale.

**Runtime:** [TensorFlow.js](https://www.tensorflow.org/js), loaded from a CDN.

No formal peer-reviewed sensitivity or specificity has been published for this exact model, so none
is claimed anywhere in this project.

**This is a research and hackathon prototype. It is not a certified medical device, and nothing it
produces constitutes a diagnosis.**
