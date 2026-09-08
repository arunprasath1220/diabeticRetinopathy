# MATLAB implementation — DR screening pipeline

This is the MATLAB/Simulink implementation of the pipeline described on the
Technical Approach slide. It is a full port of the algorithm that the browser
demo in [`js/app.js`](../js/app.js) runs, stage for stage, using the toolboxes
the slide names.

The web app is unchanged and remains the live demo. This directory is the
reference implementation: the same measurements, in the environment the project
declares.

---

## Toolbox usage

Every toolbox on the slide is used for real work, not decoration. Where a
built-in does the job better than the hand-written version in the JavaScript, the
built-in is used and the JavaScript's approach is recorded in the comment.

| Toolbox | Where it is used | What for |
|---|---|---|
| **Image Processing** | throughout | `imclose`/`imopen` with `strel('line',…)` for the directional morphology the whole detector rests on; `imreconstruct` for hysteresis and for growing the disc; `bwconncomp`/`regionprops` for regions; `adapthisteq` for CLAHE; `imboxfilt`, `imerode`, `imdilate`, `imresize`, `rgb2gray`, `bwareaopen` |
| **Computer Vision** | `renderOverlays.m` | `insertShape`, `insertText`, `insertMarker` — annotation on exact pixel coordinates, headless, so the same code runs inside a Simulink run with no display |
| **Deep Learning** | `trainClassifier.m`, `classifyImage.m`, `computeGradCAM.m` | MobileNetV2 transfer learning, `trainnet`, `predict`, and the toolbox `gradCAM` for explainability |
| **Medical Imaging** | `readFundus.m` | DICOM Ophthalmic Photography input, so PACS-attached cameras and phone-adapter rigs both work; acquisition metadata is carried into the report |
| **Statistics and ML** | `robustSigma.m`, `tailQuantile.m`, `evaluateModel.m` | `mad` for the noise floor, `prctile` for the top-hat tail, `perfcurve`/`confusionmat` for ROC-AUC and the operating point, plus calibration (ECE, Brier) and predictive entropy |
| **Simulink** | `simulink/` | the district telemedicine deployment model: arrivals, quality gate, uplink queue, inference, specialist workload |

---

## Requirements

MATLAB **R2023b or later** is the clean target. Two specific dependencies:

- `trainnet` and `minibatchpredict` in `trainClassifier.m` need R2023b+.
  `imagePretrainedNetwork` needs R2024a+; `trainClassifier.m` falls back to
  `mobilenetv2` + manual head replacement when it is absent.
- `gradCAM` needs R2021a+.

Everything else — the whole classical pipeline, which is Steps 1, 3, 4 and 5 —
runs on R2020b+ and needs only Image Processing and Statistics.

---

## Running it

```matlab
addpath('matlab');                       % so the +dr package resolves

R = dr.runPipeline('path/to/fundus.jpg', Render=true);
disp(dr.writeReport(R));
imshow(R.panels.lesionOverlay);
```

Without a trained classifier in `matlab/models/drClassifier.mat`, Steps 2 and 3a
are skipped and reported as unavailable — the lesion detector and the ICDR
estimate still run, because they are classical and do not need the network. What
is lost is the cross-check between the two, and `assessSeverity` records that
absence as a reason for caution rather than letting it pass.

To train one:

```matlab
net = dr.trainClassifier('path/to/dataset');   % dataset/no_dr, dataset/dr
```

To build and run the deployment model:

```matlab
addpath('matlab/simulink');
buildDRScreeningModel();
sim('drScreeningDeployment');
```

Tests:

```matlab
runAllTests    % from matlab/tests
```

---

## Map to the JavaScript

The two implementations are the same algorithm. This table is the correspondence
a reviewer would need to check one against the other.

| Stage | `js/app.js` | MATLAB |
|---|---|---|
| Quality metrics | `computeMetrics` | `dr.computeMetrics` |
| Quality gate | `assessQuality` | `dr.assessQuality` |
| Enhancement | `enhanceImage`, `claheLite` | `dr.enhanceImage` (`adapthisteq`) |
| Field mask | `retinaFieldMask` | `dr.retinaFieldMask` |
| Directional morphology | `directionalMorph` | `dr.directionalMorph` (`imclose`/`strel`) |
| Seeds and vessel map | `roundStructureSeeds` | `dr.roundStructureSeeds` |
| Hysteresis | `hysteresisMask` | `dr.hysteresisMask` (`imreconstruct`) |
| Optic disc | `estimateOpticDisc` | `dr.estimateOpticDisc` |
| Disc extent fill | `floodBrightRegion` | `dr.floodBrightRegion` |
| Fovea / macula | `estimateFoveaMacula` | `dr.estimateFoveaMacula` |
| Regions | `connectedComponents` | `dr.regionFeatures` (`bwconncomp`/`regionprops`) |
| Lesion detection | `detectLesionCandidates` | `dr.detectLesionCandidates` |
| Junction test | `vesselArms`, `isVesselJunction` | `dr.vesselArms`, `dr.isVesselJunction` |
| ICDR grading | `assessSeverity` | `dr.assessSeverity` |
| Classifier | `runInference` (TF.js) | `dr.classifyImage` (Deep Learning Toolbox) |
| Grad-CAM | `computeGradCAM` | `dr.computeGradCAM` |

---

## Where the two will not agree exactly, and why

The port is algorithm-for-algorithm, not bit-for-bit. These are the known
differences. None of them changes a verdict on a normal image, but all of them
are real and it is better to have them written down than to discover them during
a demo.

1. **Pixel coordinates are 1-based.** The JavaScript reports 0-based positions.
   Every reported centre differs by one pixel. Nothing downstream depends on it,
   because all the thresholds are on distances, not positions.

2. **`robustSigma` and `tailQuantile` are exact here, binned there.** The
   JavaScript computes both through histograms — bin width 0.5 grey levels for
   the median, 0.25 for the MAD — to stay a single linear pass in the browser.
   MATLAB uses `mad` and `prctile` directly. Thresholds therefore differ by up to
   about half a bin, which occasionally moves one borderline candidate.

3. **CLAHE is `adapthisteq`, not the hand-written version.** The clip limit is
   converted exactly (`L = (k-1)/(B-1) = 2.5/255`, derived in `Config.m`), and
   both use uniform distribution with bilinear tile interpolation. What differs
   is the excess-redistribution scheme, which MATLAB iterates and the JavaScript
   does in one pass. The effect is a fraction of a grey level.

4. **Line structuring elements are Bresenham.** `strel('line',…)` discretises an
   oblique line slightly differently from the JavaScript's `round(x*slope)`
   construction. At the same nominal length the two elements can differ by a
   pixel at the ends, which matters only for structures a few pixels across.

5. **Morphological borders.** Both use window truncation, which for max/min
   filters is identical to replicate padding, so these agree — with the exception
   of the line elements above at the extreme frame edge, which the retinal field
   mask excludes anyway.

6. **The classifier is a different network.** The web app loads a MobileNet via
   TensorFlow.js; this loads one trained by `dr.trainClassifier`. Steps 2 and 3a
   will not match numerically and are not expected to. Steps 1, 4 and 5 are
   deterministic image processing and should agree closely.

---

## Verification status — read this before presenting it

**The MATLAB code in this directory has not been executed.** It was written in an
environment with no MATLAB installation. What *has* been done:

- Every file passes a structural check for block and bracket balance and for
  unresolved `dr.*` calls (46 files, 0 problems).
- The algorithm it implements is the one the JavaScript runs, and *that* is
  tested: the peripapillary suppression rules were verified against a synthetic
  harness in JS, with a paired white-versus-yellow test at matched size,
  brightness and distance from the disc.
- `tests/tPeripapillary.m` mirrors those JS assertions one for one, so running
  `runAllTests` is the first thing to do on a machine with MATLAB, and a
  disagreement between the two implementations will show up there rather than
  quietly.

Treat the first run as a debugging pass, not a demo.
