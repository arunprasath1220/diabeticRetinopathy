(function(){
  "use strict";

  // ---------------------------------------------------------------------
  // CONFIG
  // ---------------------------------------------------------------------
  const MODEL_URLS = [
    "https://cdn.jsdelivr.net/gh/vbookshelf/Diabetic-Retinopathy-Analyzer@master/model_dr_2/model.json",
    "https://raw.githubusercontent.com/vbookshelf/Diabetic-Retinopathy-Analyzer/master/model_dr_2/model.json"
  ];
  const CLASS_NAMES = ["No DR", "DR present"];
  // Grad-CAM layer. conv_pw_13_relu is the last convolutional layer, but at this
  // model's 224px input it is only 7x7, so upsampling it to the display gives
  // roughly 114-pixel cells — the rectangular plateaus that make the heat map look
  // blocky and put its edges nowhere near any actual lesion. conv_pw_11_relu is the
  // deepest 14x14 layer, halving the cell size for the same kind of features.
  const CAM_LAYER_CANDIDATES = ["conv_pw_11_relu", "conv_pw_13_relu"];
  const LAST_CONV_LAYER = "conv_pw_13_relu";
  // Spatial resolution of the chosen Grad-CAM layer at a 224px input. Used only
  // to size the trim that absorbs the upsampling bleed past the aperture edge,
  // so being one step out is harmless.
  const CAM_GRID_HINT = 14;
  // How far above the retina's own typical activation a region must peak before
  // it is marked. Grad-CAM is scaled to its own maximum, so *something* always
  // reaches 1.0 — including on an image where the model found nothing. This is
  // what separates a hotspot from the top of a flat map, and it is measured
  // against the map's median absolute deviation over retina, so it follows each
  // image rather than assuming a fixed contrast.
  const CAM_NOISE_K = 3.0;
  // A map whose retinal median is already this high is uniformly warm: there is
  // no hotspot to speak of, only a ceiling, and circling its highest corner
  // would invent a localisation the model never made.
  const CAM_FLAT_MEDIAN = 0.55;
  // The cosmetic fade at the edge of the analysed field, as a fraction of the
  // trim itself. Sized this way the overlay ramps out across most of the trimmed
  // rim, so the boundary reads as a vignette rather than as a hard ring drawn
  // around the eye. Drawing only — detection uses the hard mask.
  const FIELD_FEATHER_FRAC = 0.7;
  const GAP_LAYER = "global_average_pooling2d";
  const DENSE_LAYER = "dense";
  const WORKING_MAX_DIM = 800;   // cap for the working/display canvas
  const METRIC_DIM = 512;        // standardized size for quality metrics, so thresholds are resolution-independent

  // Grad-CAM region marking: a region is circled when its activation exceeds this
  // fraction of the map's peak and it covers at least this fraction of the frame.
  // Both are display heuristics for legibility, not clinical detection thresholds.
  let BLOB_THRESHOLD = 0.50;
  let BLOB_MIN_AREA_FRAC = 0.0012;

  // Most candidates the fundus overlay will ring before it becomes unreadable.
  // The mask canvases are unaffected and always show every candidate pixel.
  const OVERLAY_CIRCLE_LIMIT = 150;
  // How many attention regions Grad-CAM may mark. The previous cap of five was
  // hiding real ones on images with widespread disease.
  let CAM_REGION_LIMIT = 12;
  // Detection sensitivity. Thresholds are multiples of a robust noise estimate,
  // so they follow each image rather than assuming one camera. Lowering DARK_K
  // finds fainter lesions and more noise; MIN_CONTRAST_K is the backstop that
  // keeps that trade from turning into speckle.
  const DARK_K = 3.5;
  const BRIGHT_K = 4.0;
  let MIN_CONTRAST_K = 3.0;
  // Absolute floors in grey levels. Without these the detector collapses on a
  // smooth image: a low noise estimate drives the adaptive threshold down to a
  // few grey levels, and normal choroidal texture then reads as lesions. A real
  // microaneurysm sits well below its background and a real exudate well above,
  // so nothing genuine is lost by refusing to look below these.
  const DARK_FLOOR = 10;
  const BRIGHT_FLOOR = 12;
  const DARK_CONTRAST_FLOOR = 9;
  const BRIGHT_CONTRAST_FLOOR = 11;
  // Cotton wool spots are a quarter to a half disc diameter across. Anything
  // small and soft-edged is far more likely to be defocus or an artefact, so
  // the class carries a size requirement rather than accepting every soft blob.
  const CWS_MIN_AREA_FRAC = 0.0004;
  // Anything sizeable this far out from the centre of the aperture is an imaging
  // artifact, not a lesion: shadows from pupil misalignment, peripapillary
  // crescents and lens flare all live at the edge of the field and can be large.
  // Genuine peripheral lesions are small, so the rule is keyed on size as well as
  // position and does not blank out the periphery.
  const EDGE_ZONE_FRAC = 0.80;
  const EDGE_ARTIFACT_AREA_FRAC = 0.0015;
  // Hysteresis. A region is found at the strict threshold and then grown out to
  // this fraction of it. Recovers cluster members and faint lesion margins that
  // a single threshold cuts off, without admitting texture, which never produces
  // a strong enough core to start a region.
  const HYST_RATIO = 0.45;
  // ---- Vessel map.
  //
  // A vessel is told from a lesion by the *spread* of the directional closing,
  // not by the size of any one response. Closing along a line fills anything
  // narrower than the element. At a vessel the element lying along it fills
  // nothing, while every element crossing it fills it completely, so the
  // responses across orientations are far apart. At a round lesion every
  // orientation fills it about equally, so they are close together. The
  // difference between the largest and smallest response — the anisotropy — is
  // therefore near zero on a lesion and large on a vessel, whatever the calibre.
  //
  // The previous map thresholded the largest response on its own and then vetoed
  // pixels whose smallest response was also large. Two absolute thresholds, and
  // neither could hold across vessel calibres: measured against a ground-truth
  // vessel tree it recovered 24.5% of vessel pixels and 3% of the thin ones, and
  // produced 755 disconnected fragments where a vascular tree should be nearly
  // one object. Three quarters of the vasculature was simply not being
  // suppressed, which is where the marks on healthy retina were coming from.
  //
  // Comparing the two responses to each other instead of to a fixed number is
  // scale-free, so one rule covers a thin peripheral capillary and a wide vein
  // at the disc alike.
  const VESSEL_ELONGATION = 1.5;
  // Hysteresis quantiles on the anisotropy map. A vessel is confidently found
  // somewhere along its length and then followed outward; this is what turns
  // fragments into a tree and what recovers the faint thin vessels, which are
  // never strong enough to clear a single global threshold but are always
  // attached to something that is.
  const VESSEL_SEED_Q = 0.95;
  const VESSEL_GROW_Q = 0.80;
  // Half-length of the linear structuring element used to separate lesions from
  // vessels. Round structures up to about twice this across are filled by a
  // closing in every direction; a vessel is longer than it in the direction it
  // runs and survives. Too small and large hemorrhages are only outlined; too
  // large and short vessel segments start to read as lesions.
  const LINEAR_SE_FRAC = 0.014;
  // Orientations tested. The angular gap sets how far a vessel can lean away
  // from the nearest element before that element walks off it; longer elements
  // need finer sampling to stay on a vessel.
  const LINEAR_ORIENTATIONS = 12;
  // Seeding thresholds on the directional top-hat.
  //
  // The adaptive term is anchored to an upper quantile of the map rather than to
  // a median absolute deviation, because MAD is the wrong statistic for this map
  // and was silently doing nothing at all. A top-hat is non-negative by
  // construction and more than half the retina responds at exactly zero, so the
  // median sits on zero and the MAD measures the width of that zero atom rather
  // than the noise. Measured on synthetic fundus images it comes out at about
  // 0.56 grey levels, putting the adaptive threshold at 1.7 while the floor was
  // 8 — so the floor decided every seed on every image and the claim that
  // sensitivity followed the image was false. That is why the exposed control
  // behaved like an on/off switch: it was the only thing connected to anything.
  // An upper quantile sits in the tail where the signal actually is, so the
  // threshold now tracks the camera, the exposure and the enhancement.
  const SEED_TAIL_Q = 0.97;
  // Backstops in grey levels, for an image whose tail estimate collapses — a
  // near-uniform frame, or one where the retina barely fills the aperture.
  let SEED_FLOOR_DARK = 8;
  let SEED_FLOOR_BRIGHT = 10;
  // ---- Scale saturation, the test that separates lesions from vessel geometry.
  //
  // Amplitude cannot do it. Measured over the retina, the top-hat response of a
  // healthy fundus and of one carrying seventeen planted lesions are the same
  // distribution out to the 99th percentile (15.1 vs 16.6 grey levels); they
  // separate only past the 99.9th. So no threshold on response strength — which
  // is the only axis the detector previously had — can tell a microaneurysm from
  // a vessel crossing, a bifurcation or the apex of a bend. Raising the floor
  // until the false marks disappear also takes the real lesions: at a floor of
  // 20 the healthy retina still seeded 0.54% of its area while lesion recall had
  // already fallen to 67%.
  //
  // What does separate them is boundedness. A lesion is a blob of finite width:
  // once the structuring element is longer than the blob is wide, the closing
  // already fills it completely, and making the element longer adds nothing. A
  // vessel bend, a crossing, or a patch of choroidal mottling is not bounded —
  // it continues past the element — so a longer element bridges more of it and
  // the response keeps climbing. Measuring the same top-hat at two element
  // lengths and taking the ratio therefore reads a property that neither the
  // amplitude nor the shape of a single response can see.
  //
  // Measured on candidate regions: true lesions have a growth ratio of 1.00
  // (quartiles 1.00-1.01) — completely saturated. False candidates sit at
  // 1.46-1.57, and separate cleanly enough that the cut can be made close to 1.
  //
  // The test is applied on a ladder of element lengths rather than a single pair,
  // and each region is judged at the rung that matches its own size. A single
  // pair leaves a hole: a region wider than the short element has not saturated
  // yet, so the test cannot speak for it and it must be exempted — and on a
  // healthy retina that exemption is exactly where the survivors collect. Every
  // false mark left on a healthy retina after the single-pair version was
  // dot/blot sized, sitting in that gap. Judging a region against elements
  // scaled to itself closes it: a blob of any size saturates at its own scale,
  // while vessel geometry keeps growing at every scale.
  //
  // Introduced on its own, end to end over seven synthetic retinas against the
  // same detector without it: false marks on a healthy retina fall from 15.6 per
  // image to 7.9, false marks on a diseased one from 14.9 to 6.7, while lesion
  // recall RISES from 72% to 76%. Precision and recall improve together because
  // the test is not a threshold trade — it removes a class of structure rather
  // than raising a bar everything has to clear. The cost is three directional
  // passes per polarity instead of one.
  //
  // With the vessel map since repaired, the two together take false marks to 3.0
  // per healthy image and 2.4 per diseased one, at 68% lesion recall and 86%
  // exudate recall: precision 45% -> 83%, recall 72% -> 68%. The recall is spent
  // on lesions lying on the vasculature, which is the same place the false marks
  // were coming from.
  //
  // It degrades on genuinely noisy frames: at a noise level of 7 grey levels
  // false marks climb back to 14 per healthy image and recall drops to 46%,
  // because noise breaks a smooth structure into pieces that each look bounded.
  // Below about 5 grey levels it is stable and needs no retuning per camera,
  // which an amplitude threshold does not manage.
  const SATURATION_SE_MULT = 2.0;
  const SATURATION_LADDER = 3;              // seL, 2*seL, 4*seL
  // The longer passes only need the size of the response, not a fine reading of
  // which way each vessel runs, so they are sampled at half the angular
  // resolution. Measured to change nothing and to halve the added cost.
  const SATURATION_ORIENTATIONS = 6;
  // Re-tuned once the vessel map was fixed. The two tests overlap: most of what
  // the saturation test used to be the only defence against is now suppressed as
  // vessel before it ever becomes a candidate, so the shape test no longer has to
  // be strict enough to catch it alone. Measured against the repaired vessel map,
  // loosening the cut from 1.05 to 1.10 recovers three lesions per three retinas
  // for one extra false mark; past 1.15 the false marks climb without buying any
  // further recall.
  let SATURATION_MAX_GROWTH = 1.10;
  // A region is only discarded as vessel when this much of it lies on the vessel
  // map. Set too low, lesions touching a vessel are lost with it.
  const VESSEL_OVERLAP_REJECT = 0.78;
  // Broad, smooth darkening is anatomy or shadow, not a lesion. The macula is
  // the main case: it is genuinely darker than surrounding retina, so a detector
  // measuring darkness against a wide background will flag it every time.
  const SMOOTH_GRADIENT_RATIO = 0.55;
  const MACULA_SMOOTH_AREA_FRAC = 0.08;
  const SMOOTH_BROAD_AREA_FRAC = 0.004;
  // ---- Peripapillary bright suppression.
  //
  // The optic disc is the brightest thing in a normal fundus, so every bright
  // candidate it produces is a false exudate. That was already meant to be
  // handled by rejecting anything centred within 1.15 disc radii — but only as
  // well as the radius was known, and the radius was the weak link.
  //
  // The radius came from the area of the plateau surviving a cut at 65% of the
  // way from the *global* retinal mean up to the disc's own blurred peak. That is
  // a fixed height on a soft-edged structure, and it consistently lands inside
  // the rim: the plateau it leaves is the bright core, not the disc. On the
  // synthetic scenes in this fix's harness it read 47 px for a 50 px disc and
  // 50 px for a 52 px one — a few per cent small, which sounds harmless until it
  // is multiplied by 1.15 and asked to reach past the rim. It does not. The rim
  // itself, a scleral crescent beside it and the peripapillary reflex all sit in
  // the gap between where the circle stops and where the disc actually ends, and
  // every one of them is bright, round and sharply bounded — which is the exact
  // description of a hard exudate. So they were reported as one: a scatter of
  // bright marks ringing a disc the detector believed it had already excluded.
  //
  // Two things follow, and both are needed:
  //
  //   1. Measure the disc's extent instead of inferring it from a plateau area.
  //      A flood fill from the disc centre, cut at a level referred to the
  //      *peripapillary* background rather than the global mean, follows the
  //      structure that is actually there. It is run twice: once at a high level
  //      for the disc proper — the radius that gets drawn and that the existing
  //      rejection circle uses — and once lower for the halo, disc plus crescent
  //      plus reflex, which is the tissue a bright candidate must not be made of.
  //
  //   2. Judge what is left on colour. Beyond the halo there is still nerve-fibre
  //      reflex and atrophic mottling, and no amount of geometry separates those
  //      from an exudate, because in shape, size and contrast they are the same
  //      thing. In colour they are not: exudate is lipid and reads yellow, while
  //      disc tissue, sclera and reflex read white or grey. The reference for
  //      "white" is read off the disc in the same image rather than fixed in
  //      advance, so the test survives any white balance.
  //
  // Both rules are confined to bright candidates inside the peripapillary
  // annulus. Nothing dark is touched, because nothing about the disc is dark and
  // a hemorrhage beside it is a real finding; nothing in the macula or the
  // periphery is touched at all.

  // How far out from the disc centre the disc can still be the explanation for a
  // bright mark, in disc radii. Peripapillary atrophy rarely reaches past this,
  // and beyond it the colour reference stops being local enough to trust.
  const PERIPAPILLARY_MULT = 2.5;
  // Level, on the scale from peripapillary background to the disc's own plateau,
  // at which the disc proper is called to end. Half way is where the half-maximum
  // contour of a blurred edge sits, which is the edge itself.
  const DISC_EDGE_LEVEL = 0.50;
  // ...and the lower level that also takes in the crescent and the reflex. The
  // ground between the two levels is the halo.
  const DISC_HALO_LEVEL = 0.28;
  // The fill is bounded so a washed-out frame cannot let it escape into open
  // retina. Hitting a bound abandons the measurement rather than trusting it, and
  // the old plateau estimate stands — a refusal is worth more than a wrong disc.
  const DISC_FILL_MAX_MULT = 3.2;
  // How much of a bright candidate must lie on measured disc tissue before the
  // disc is accepted as its explanation. Deliberately not a majority: a real
  // exudate does not overlap the disc at all, so anything with a third of itself
  // built out of disc pixels is a piece of the disc that grew outward.
  const DISC_TISSUE_OVERLAP = 0.33;
  // Where a peripapillary candidate's blue fraction has to sit, on the scale from
  // surrounding retina (0) to disc white (1), before colour calls it disc tissue.
  // Above half, so a mark has to look more like the disc than like the retina it
  // lies on; a yellow exudate lands well below.
  const DISC_COLOUR_LEVEL = 0.55;
  // Below this separation between disc and retinal blue fraction the reference is
  // degenerate — a monochrome frame, or one with a heavy colour cast — and the
  // colour test is skipped rather than answered on noise.
  const DISC_COLOUR_MIN_SEPARATION = 0.02;

  // Above this many candidates, the detector is almost certainly firing on noise
  // rather than lesions, and the UI says so instead of presenting a tidy count.
  const NOISE_SUSPICION_COUNT = 400;

  const QUALITY_THRESHOLDS = {
    minFocusVar: 15,
    enhanceFocusVar: 40,
    minBrightMean: 40,
    maxBrightMean: 235,
    enhanceBrightLow: 60,
    enhanceBrightHigh: 200,
    minFov: 0.35,
    enhanceFov: 0.5
  };

  const state = {
    model: null,
    modelReady: false,
    modelLoadAttempted: false,
    activationModel: null,
    activationModelLayer: null,
    camLayerName: null,
    workingCanvas: null,
    procCanvas: null,
    lastQuality: null,
    lastGrading: null,
    lastGradcamDesc: null,
    lastCam: null,
    lastAnatomy: null,
    lastFindings: null,
    lastLesions: null,
    lastSeverity: null
  };

  // ---------------------------------------------------------------------
  // SMALL HELPERS
  // ---------------------------------------------------------------------
  function $(sel){ return document.querySelector(sel); }
  // Lets the browser paint a status line before a long synchronous stage.
  function yieldToBrowser(){ return new Promise(r => setTimeout(r, 0)); }
  function clamp(v,lo,hi){ return Math.min(hi, Math.max(lo, v)); }
  function clampInt(v,lo,hi){ return Math.min(hi, Math.max(lo, v|0)); }
  function showStep(id){ const el = document.getElementById(id); if (el) el.classList.add("visible"); }
  function fmt(n, d){ return Number(n).toLocaleString(undefined, {minimumFractionDigits:d, maximumFractionDigits:d}); }

  function updateModelStatus(kind, text){
    const el = $("#model-status");
    el.textContent = text;
    el.className = kind;
    const prog = $("#model-progress");
    if (kind === "ready" || kind === "error"){ prog.remove(); }
  }

  // ---------------------------------------------------------------------
  // MODEL LOADING
  // ---------------------------------------------------------------------
  async function loadModel(){
    for (const url of MODEL_URLS){
      try{
        updateModelStatus("loading", "Loading classifier model (MobileNet, TensorFlow.js)…");
        const m = await tf.loadLayersModel(url);
        // warm-up pass so the first real prediction isn't slowed by lazy graph compilation
        tf.tidy(() => { m.predict(tf.zeros([1,224,224,3])); });
        state.model = m;
        state.modelReady = true;
        state.modelLoadAttempted = true;
        updateModelStatus("ready", "Model ready — MobileNet binary DR classifier loaded client-side.");
        $("#run-button").disabled = $("#file-input").files.length === 0;
        return;
      } catch(err){
        console.warn("Model load failed from", url, err);
      }
    }
    state.modelReady = false;
    state.modelLoadAttempted = true;
    updateModelStatus("error", "Classifier unavailable — demo mode. The model could not be downloaded (check your network connection). No DR grade or Grad-CAM will be shown, since generating placeholder numbers would misrepresent the model.");
    $("#run-button").disabled = $("#file-input").files.length === 0;
  }

  // ---------------------------------------------------------------------
  // IMAGE LOADING / CANVAS SETUP
  // ---------------------------------------------------------------------
  function loadImageFile(file){
    return new Promise((resolve, reject) => {
      const img = new Image();
      const url = URL.createObjectURL(file);
      img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = (e) => { URL.revokeObjectURL(url); reject(e); };
      img.src = url;
    });
  }

  function drawToCanvas(img, canvas, maxDim){
    let w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
    const scale = Math.min(1, maxDim / Math.max(w,h));
    canvas.width = Math.max(1, Math.round(w*scale));
    canvas.height = Math.max(1, Math.round(h*scale));
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  function copyCanvas(src){
    const c = document.createElement("canvas");
    c.width = src.width; c.height = src.height;
    c.getContext("2d").drawImage(src,0,0);
    return c;
  }

  function resizeCanvasCopy(src, maxDim){
    const scale = Math.min(1, maxDim / Math.max(src.width, src.height));
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(src.width*scale));
    c.height = Math.max(1, Math.round(src.height*scale));
    c.getContext("2d").drawImage(src, 0, 0, c.width, c.height);
    return c;
  }

  // ---------------------------------------------------------------------
  // QUALITY METRICS (real pixel computation)
  // ---------------------------------------------------------------------
  function computeMetrics(sourceCanvas){
    const mCanvas = resizeCanvasCopy(sourceCanvas, METRIC_DIM);
    const w = mCanvas.width, h = mCanvas.height;
    const d = mCanvas.getContext("2d").getImageData(0,0,w,h).data;
    const gray = new Float32Array(w*h);
    for (let i=0, p=0; i<d.length; i+=4, p++){
      gray[p] = 0.299*d[i] + 0.587*d[i+1] + 0.114*d[i+2];
    }

    // Laplacian-variance focus measure
    const lap = new Float32Array(w*h);
    let lapSum = 0, n = 0;
    for (let y=1; y<h-1; y++){
      for (let x=1; x<w-1; x++){
        const idx = y*w+x;
        const v = gray[idx-1] + gray[idx+1] + gray[idx-w] + gray[idx+w] - 4*gray[idx];
        lap[idx] = v; lapSum += v; n++;
      }
    }
    const lapMean = n ? lapSum/n : 0;
    let lapSumSq = 0;
    for (let y=1; y<h-1; y++){
      for (let x=1; x<w-1; x++){
        const idx = y*w+x;
        const diff = lap[idx]-lapMean;
        lapSumSq += diff*diff;
      }
    }
    const focusVar = n ? lapSumSq/n : 0;

    // Brightness stats
    let sum = 0;
    for (let i=0;i<gray.length;i++) sum += gray[i];
    const brightMean = sum/gray.length;
    let sumSq = 0;
    for (let i=0;i<gray.length;i++){ const diff = gray[i]-brightMean; sumSq += diff*diff; }
    const brightStd = Math.sqrt(sumSq/gray.length);

    // Field-of-view coverage: fraction of frame that isn't near-black border
    let nonBlack = 0;
    for (let i=0;i<gray.length;i++) if (gray[i] > 15) nonBlack++;
    const fov = nonBlack/gray.length;

    return { focusVar, brightMean, brightStd, fov };
  }

  function assessQuality(metrics){
    const T = QUALITY_THRESHOLDS;
    const reasons = [];
    if (metrics.fov < T.minFov){
      return { verdict:"reject", reason:`Insufficient retinal field of view (only ${(metrics.fov*100).toFixed(0)}% of the frame is non-black) — recentre the eye in the camera and recapture.` };
    }
    if (metrics.brightMean < T.minBrightMean){
      return { verdict:"reject", reason:`Mean brightness too low (${metrics.brightMean.toFixed(1)}/255) — recapture with more illumination.` };
    }
    if (metrics.brightMean > T.maxBrightMean){
      return { verdict:"reject", reason:`Image overexposed (mean brightness ${metrics.brightMean.toFixed(1)}/255) — reduce flash intensity and recapture.` };
    }
    if (metrics.focusVar < T.minFocusVar){
      return { verdict:"reject", reason:`Image too blurry (focus score ${metrics.focusVar.toFixed(1)}) — hold the camera steady and refocus before recapture.` };
    }

    const needsEnhance = metrics.focusVar < T.enhanceFocusVar
      || metrics.brightMean < T.enhanceBrightLow
      || metrics.brightMean > T.enhanceBrightHigh
      || metrics.fov < T.enhanceFov;

    if (needsEnhance){
      if (metrics.focusVar < T.enhanceFocusVar) reasons.push("borderline focus");
      if (metrics.brightMean < T.enhanceBrightLow || metrics.brightMean > T.enhanceBrightHigh) reasons.push("borderline exposure");
      if (metrics.fov < T.enhanceFov) reasons.push("limited field of view");
      return { verdict:"enhance", reasons };
    }
    return { verdict:"ok" };
  }

  // ---------------------------------------------------------------------
  // ENHANCEMENT: separable box blur, illumination normalization, CLAHE-lite
  // ---------------------------------------------------------------------
  function boxBlur(src, w, h, radius){
    if (radius <= 0) return src.slice();
    const tmp = new Float32Array(w*h);
    const out = new Float32Array(w*h);
    const win = 2*radius+1;
    const inv = 1/win;
    const wLast = w-1, hLast = h-1;

    for (let y=0; y<h; y++){
      const rowOff = y*w;
      let sum = 0;
      for (let k=-radius; k<=radius; k++){
        let xx = k; if (xx < 0) xx = 0; else if (xx > wLast) xx = wLast;
        sum += src[rowOff+xx];
      }
      tmp[rowOff] = sum*inv;
      for (let x=1; x<w; x++){
        let a = x+radius; if (a > wLast) a = wLast;
        let b = x-radius-1; if (b < 0) b = 0;
        sum += src[rowOff+a] - src[rowOff+b];
        tmp[rowOff+x] = sum*inv;
      }
    }

    for (let x=0; x<w; x++){
      let sum = 0;
      for (let k=-radius; k<=radius; k++){
        let yy = k; if (yy < 0) yy = 0; else if (yy > hLast) yy = hLast;
        sum += tmp[yy*w+x];
      }
      out[x] = sum*inv;
      for (let y=1; y<h; y++){
        let a = y+radius; if (a > hLast) a = hLast;
        let b = y-radius-1; if (b < 0) b = 0;
        sum += tmp[a*w+x] - tmp[b*w+x];
        out[y*w+x] = sum*inv;
      }
    }
    return out;
  }

  function illuminationNormalize(Y, w, h){
    const radius = Math.max(4, Math.round(Math.min(w,h)/10));
    const blurred = boxBlur(Y, w, h, radius);
    let globalMean = 0;
    for (let i=0;i<Y.length;i++) globalMean += Y[i];
    globalMean /= Y.length;
    const out = new Float32Array(Y.length);
    for (let i=0;i<Y.length;i++){
      out[i] = clamp(Y[i] - blurred[i] + globalMean, 0, 255);
    }
    return out;
  }

  function claheLite(Y, w, h, tilesX, tilesY){
    const tileW = Math.ceil(w/tilesX), tileH = Math.ceil(h/tilesY);
    const maps = [];
    for (let ty=0; ty<tilesY; ty++){
      for (let tx=0; tx<tilesX; tx++){
        const x0 = tx*tileW, y0 = ty*tileH;
        const x1 = Math.min(w, x0+tileW), y1 = Math.min(h, y0+tileH);
        const hist = new Float32Array(256);
        let count = 0;
        for (let y=y0; y<y1; y++){
          for (let x=x0; x<x1; x++){
            const v = clampInt(Math.round(Y[y*w+x]), 0, 255);
            hist[v]++; count++;
          }
        }
        if (count === 0) count = 1;
        const avg = count/256;
        const clipLimit = avg*3.5;
        let excess = 0;
        for (let i=0;i<256;i++){
          if (hist[i] > clipLimit){ excess += hist[i]-clipLimit; hist[i] = clipLimit; }
        }
        const redist = excess/256;
        for (let i=0;i<256;i++) hist[i] += redist;
        const map = new Float32Array(256);
        let cum = 0;
        for (let i=0;i<256;i++){ cum += hist[i]; map[i] = clamp((cum/count)*255, 0, 255); }
        maps.push(map);
      }
    }
    const out = new Float32Array(w*h);
    for (let y=0; y<h; y++){
      for (let x=0; x<w; x++){
        let fx = (x - tileW/2)/tileW;
        let fy = (y - tileH/2)/tileH;
        let tx0 = Math.floor(fx), ty0 = Math.floor(fy);
        let tx1 = tx0+1, ty1 = ty0+1;
        const wx = fx - tx0, wy = fy - ty0;
        tx0 = clampInt(tx0, 0, tilesX-1); tx1 = clampInt(tx1, 0, tilesX-1);
        ty0 = clampInt(ty0, 0, tilesY-1); ty1 = clampInt(ty1, 0, tilesY-1);
        const v = clampInt(Math.round(Y[y*w+x]), 0, 255);
        const m00 = maps[ty0*tilesX+tx0][v];
        const m10 = maps[ty0*tilesX+tx1][v];
        const m01 = maps[ty1*tilesX+tx0][v];
        const m11 = maps[ty1*tilesX+tx1][v];
        const top = m00*(1-wx) + m10*wx;
        const bot = m01*(1-wx) + m11*wx;
        out[y*w+x] = top*(1-wy) + bot*wy;
      }
    }
    return out;
  }

  function enhanceImage(canvas){
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    const imgData = ctx.getImageData(0,0,w,h);
    const d = imgData.data;
    const Y = new Float32Array(w*h), Cb = new Float32Array(w*h), Cr = new Float32Array(w*h);
    for (let i=0, p=0; i<d.length; i+=4, p++){
      const R=d[i], G=d[i+1], B=d[i+2];
      Y[p] = 0.299*R + 0.587*G + 0.114*B;
      Cb[p] = -0.168736*R - 0.331264*G + 0.5*B + 128;
      Cr[p] = 0.5*R - 0.418688*G - 0.081312*B + 128;
    }
    // light denoise: blend 30% of a radius-1 box blur
    const den = boxBlur(Y, w, h, 1);
    for (let i=0;i<Y.length;i++) Y[i] = Y[i]*0.7 + den[i]*0.3;
    // illumination flattening
    const illum = illuminationNormalize(Y, w, h);
    // local contrast (CLAHE-lite)
    const finalY = claheLite(illum, w, h, 8, 8);
    for (let i=0, p=0; i<d.length; i+=4, p++){
      const Yv = finalY[p], Cbv = Cb[p], Crv = Cr[p];
      d[i]   = clamp(Yv + 1.402*(Crv-128), 0, 255);
      d[i+1] = clamp(Yv - 0.344136*(Cbv-128) - 0.714136*(Crv-128), 0, 255);
      d[i+2] = clamp(Yv + 1.772*(Cbv-128), 0, 255);
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // ---------------------------------------------------------------------
  // INFERENCE
  // ---------------------------------------------------------------------
  async function runInference(canvas){
    const t0 = performance.now();
    const logits = tf.tidy(() => {
      let t = tf.browser.fromPixels(canvas);
      t = tf.image.resizeNearestNeighbor(t, [224,224]).toFloat();
      const offset = tf.scalar(127.5);
      const normalized = t.sub(offset).div(offset).expandDims(0);
      return state.model.predict(normalized);
    });
    const probs = Array.from(await logits.data());
    logits.dispose();
    const elapsedMs = performance.now() - t0;
    const predClass = probs[1] >= probs[0] ? 1 : 0;
    return { probs, predClass, elapsedMs };
  }

  // ---------------------------------------------------------------------
  // GRAD-CAM
  // ---------------------------------------------------------------------
  // Picks the deepest available layer from the candidate list.
  function camLayerIndex(){
    const names = state.model.layers.map(l => l.name);
    for (const cand of CAM_LAYER_CANDIDATES){
      const i = names.indexOf(cand);
      if (i >= 0) return i;
    }
    return names.indexOf(LAST_CONV_LAYER);
  }

  async function computeGradCAM(canvas, classIndex){
    const layerIdx = camLayerIndex();
    if (layerIdx < 0) throw new Error("no usable convolutional layer for Grad-CAM");
    const convLayer = state.model.layers[layerIdx];
    state.camLayerName = convLayer.name;

    if (!state.activationModel || state.activationModelLayer !== convLayer.name){
      if (state.activationModel) state.activationModel = null;
      state.activationModel = tf.model({ inputs: state.model.inputs, outputs: convLayer.output });
      state.activationModelLayer = convLayer.name;
    }

    const inputTensor = tf.tidy(() => {
      let t = tf.browser.fromPixels(canvas);
      t = tf.image.resizeNearestNeighbor(t, [224,224]).toFloat();
      const offset = tf.scalar(127.5);
      return t.sub(offset).div(offset).expandDims(0);
    });

    const convOutput = state.activationModel.predict(inputTensor);
    const varName = "gradcam_conv_var_" + Date.now() + "_" + Math.random().toString(36).slice(2);
    const convVar = tf.variable(convOutput, true, varName);

    // Everything after the chosen layer is applied in order. MobileNet has no skip
    // connections, so the remainder of the network is a straight chain and can be
    // replayed this way from any point in it.
    const { grads } = tf.variableGrads(() => {
      let x = convVar;
      for (let i=layerIdx+1; i<state.model.layers.length; i++){
        x = state.model.layers[i].apply(x);
      }
      const flat = x.reshape([2]);
      return tf.unstack(flat)[classIndex];
    }, [convVar]);

    const gradTensor = grads[varName];

    // Deliberately NOT normalised here. Dividing by the global max inside the
    // graph is what put the heat map in the black surround: the network sees a
    // 224x224 square, most of which on a fundus photograph is not retina at all,
    // and on an image with nothing to find the largest response left after the
    // ReLU can easily be a corner of that surround. Scaled to its own maximum
    // that corner becomes 1.0 and gets circled as the model's strongest
    // evidence. The raw map is carried out instead and scaled against the
    // retina only, below.
    const resized = tf.tidy(() => {
      const pooledGrad = gradTensor.mean([0,1,2]);           // per-channel weight
      const convSq = convVar.squeeze([0]);                   // [H,W,C]
      const weighted = convSq.mul(pooledGrad);
      let cam = weighted.sum(-1);                            // [H,W]
      cam = cam.relu();
      return tf.image.resizeBilinear(cam.expandDims(-1), [canvas.height, canvas.width]); // [H,W,1]
    });

    const raw = await resized.data();
    const h = resized.shape[0], w = resized.shape[1];

    inputTensor.dispose();
    convOutput.dispose();
    gradTensor.dispose();
    convVar.dispose();
    resized.dispose();

    const conf = confineCamToRetina(canvas, raw, w, h);
    return { camArray: conf.camArray, h, w, inside: conf.inside, feather: conf.feather,
             field: conf.field, insideCount: conf.insideCount, rawMax: conf.rawMax, stats: conf.stats };
  }

  // Restricts the activation map to the retina and rescales it there.
  //
  // Two separate things go wrong without this, and they compound. First, the
  // black surround outside the camera aperture is not retina, so any activation
  // over it is meaningless by construction — yet it is included in the maximum
  // the map is divided by, and on a clean image it can *be* that maximum.
  // Second, the map arrives as a bilinear upsample of a 14x14 grid, so each cell
  // spans about a fourteenth of the frame and activation from the last ring of
  // retina bleeds well past the aperture edge. Masking alone would therefore
  // still leave a bright rim just inside the border. The mask is eroded by half
  // a cell to absorb that bleed before anything is measured.
  function confineCamToRetina(canvas, raw, w, h){
    const { lum } = imageChannels(canvas);
    const ap = apertureGeometry(lum, w, h);
    let inside = retinaFieldMask(lum, w, h, 0.05);
    // Half a Grad-CAM cell at this display size. Below the CAM's own resolution
    // there is nothing to be gained by a finer trim, and above it genuine
    // peripheral evidence starts being discarded.
    const bleed = Math.max(2, Math.round(Math.min(w,h)/(CAM_GRID_HINT*2)));
    inside = erodeMask(inside, w, h, bleed);

    const n = w*h;
    let insideCount = 0, rawMax = 0;
    for (let i=0;i<n;i++){
      if (!inside[i]) continue;
      insideCount++;
      if (raw[i] > rawMax) rawMax = raw[i];
    }

    // A soft copy of the same boundary, for drawing only. Painting an overlay
    // through the hard mask leaves a step at its edge — analysed retina tinted,
    // the trimmed rim not — and that step reads as a rendering fault rather than
    // as the edge of the analysed field. A box blur of the mask gives a weight
    // that falls from one to zero across a few pixels, so the overlay fades out
    // instead. Detection still uses the hard mask: what is measured and what is
    // painted are deliberately not the same thing.
    const feather = boxBlur(Float32Array.from(inside), w, h,
      clampInt(Math.round(bleed*FIELD_FEATHER_FRAC), 3, 24));
    // Where that boundary lies, so the panels can draw it. The rim between this
    // circle and the aperture is real retina that is deliberately not judged:
    // a Grad-CAM cell straddling the edge is computed from a receptive field
    // that is mostly black surround, so its value says nothing about retina.
    // Showing the line turns an unexplained ring into a stated exclusion.
    const field = { cx: ap.cx, cy: ap.cy, r: Math.max(1, ap.R*0.955 - bleed) };

    const camArray = new Float32Array(n);
    if (insideCount === 0 || rawMax <= 1e-12){
      return { camArray, inside, feather, field, insideCount, rawMax, stats: { median:0, mad:0 } };
    }
    const invMax = 1/rawMax;
    for (let i=0;i<n;i++) camArray[i] = inside[i] ? raw[i]*invMax : 0;

    return { camArray, inside, feather, field, insideCount, rawMax, stats: camFieldStats(camArray, inside, n) };
  }

  // Median and median absolute deviation of the normalised map over retina only,
  // by histogram so it stays a linear pass. These are what tell a focal hotspot
  // from a map that is simply bright everywhere: the peak is 1.0 either way, but
  // only in the first case is the typical retinal pixel far below it.
  function camFieldStats(cam, inside, n){
    const BINS = 512;
    const hist = new Int32Array(BINS);
    let count = 0;
    const step = n > 300000 ? 2 : 1;
    for (let i=0;i<n;i+=step){
      if (!inside[i]) continue;
      let b = (cam[i]*BINS)|0;
      if (b >= BINS) b = BINS-1; else if (b < 0) b = 0;
      hist[b]++; count++;
    }
    if (!count) return { median:0, mad:0 };
    const half = count/2;
    let cum = 0, median = 0;
    for (let b=0;b<BINS;b++){
      cum += hist[b];
      if (cum >= half){ median = (b+0.5)/BINS; break; }
    }
    const ahist = new Int32Array(BINS);
    for (let i=0;i<n;i+=step){
      if (!inside[i]) continue;
      let a = cam[i] - median;
      if (a < 0) a = -a;
      let b = (a*BINS)|0;
      if (b >= BINS) b = BINS-1;
      ahist[b]++;
    }
    let acum = 0, mad = 1;
    for (let b=0;b<BINS;b++){
      acum += ahist[b];
      if (acum >= half){ mad = (b+0.5)/BINS; break; }
    }
    return { median, mad: 1.4826*mad };
  }

  // Every statistic here is taken over retina only. Measured over the whole
  // frame instead, the black surround contributes a large block of zeros that
  // inflates the concentration figure — a map spread evenly across the whole
  // retina still looks focal simply because the retina is a minority of the
  // frame — and the peak it reports can land outside the eye entirely.
  function describeGradCAM(cam, anatomy){
    const { camArray, h, w, inside, insideCount, stats } = cam;
    const n = w*h;
    let maxVal = -Infinity, maxIdx = -1, sum = 0;
    const vals = new Float32Array(insideCount || 0);
    let k = 0;
    for (let i=0;i<n;i++){
      if (inside && !inside[i]) continue;
      const v = camArray[i];
      sum += v;
      if (k < vals.length) vals[k++] = v;
      if (v > maxVal){ maxVal = v; maxIdx = i; }
    }
    if (maxIdx < 0) maxIdx = 0;
    const py = Math.floor(maxIdx/w), px = maxIdx % w;
    const region = anatomy
      ? quadrantLabel(px, py, anatomy.disc, anatomy.nasalSide)
      : (py < h*0.5 ? "superior" : "inferior") + " frame";

    const sorted = Array.prototype.slice.call(vals.subarray(0, k)).sort((a,b)=>b-a);
    const topN = Math.max(1, Math.round(sorted.length*0.1));
    let topSum = 0;
    for (let i=0;i<topN && i<sorted.length;i++) topSum += sorted[i];
    const concentration = sum > 1e-8 ? topSum/sum : 0;
    const focal = concentration > 0.35;
    const flat = !!(stats && stats.median >= CAM_FLAT_MEDIAN);

    return { region, focal, flat, concentration, median: stats ? stats.median : 0, peakX: px, peakY: py };
  }

  // ---------------------------------------------------------------------
  // ANATOMICAL LANDMARK ESTIMATION (heuristic — disclosed as such in the UI)
  //
  // Optic disc: brightest large region after a heavy blur. The blur radius is
  // chosen so small bright lesions (hard exudates) and specular reflections are
  // smoothed away, while the disc — a much larger contiguous bright area —
  // survives as the global maximum.
  // ---------------------------------------------------------------------
  // Shared channel extraction and retinal field mask.
  function imageChannels(canvas){
    const w = canvas.width, h = canvas.height, n = w*h;
    const d = canvas.getContext("2d").getImageData(0,0,w,h).data;
    const lum = new Float32Array(n), green = new Float32Array(n);
    // Red and blue are carried too. Nothing in the morphology uses them — that
    // all runs on green and luminance — but telling lipid exudate from disc
    // tissue is a question about hue, not about shape, and needs all three.
    const red = new Float32Array(n), blue = new Float32Array(n);
    for (let i=0, p=0; i<d.length; i+=4, p++){
      green[p] = d[i+1];                                     // strongest blood contrast
      red[p] = d[i];
      blue[p] = d[i+2];
      lum[p] = 0.299*d[i] + 0.587*d[i+1] + 0.114*d[i+2];
    }
    return { w, h, n, lum, green, red, blue };
  }

  // Centre and radius of the camera aperture, from the extent of the lit area.
  function apertureGeometry(lum, w, h){
    let hi = 0;
    for (let i=0;i<w*h;i++) if (lum[i] > hi) hi = lum[i];
    const thr = Math.max(14, hi*0.20);
    let minX = w, maxX = -1, minY = h, maxY = -1;
    for (let y=0; y<h; y++){
      for (let x=0; x<w; x++){
        if (lum[y*w+x] <= thr) continue;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    if (maxX < 0) return { cx: w/2, cy: h/2, R: Math.min(w,h)/2 };
    return {
      cx: (minX+maxX)/2, cy: (minY+maxY)/2,
      R: Math.max(1, Math.max(maxX-minX, maxY-minY)/2)
    };
  }

  // The usable retina, found by fitting the camera aperture rather than trimming a
  // fixed margin off whatever is brighter than a fixed number.
  //
  // Both of those shortcuts failed on real images. A fixed brightness cutoff keeps
  // the vignetted rim, which is genuine retina but far darker than the rest, so it
  // reads as one enormous dark lesion — the crescent that appeared beside the disc.
  // A fixed erosion cannot reach it either, because the vignette is much wider than
  // the margin. Fitting the aperture and working inside a fraction of its radius
  // handles both, and adapts to how much of the frame the retina fills. The
  // fraction kept is deliberately generous: trimming a tenth of the radius was
  // measured to discard genuine peripheral lesions, so large rim artifacts are
  // dealt with by a size-keyed rule during classification instead.
  function retinaFieldMask(lum, w, h, erodeFrac){
    const n = w*h;
    let hi = 0;
    for (let i=0;i<n;i++) if (lum[i] > hi) hi = lum[i];

    // Two different thresholds, for two different jobs.
    //
    // Fitting the aperture wants a firm cutoff, so the dim vignetted rim does not
    // stretch the circle. Deciding which pixels are *inside* that aperture wants a
    // far lower one, because a hemorrhage is dark: judged at the fitting cutoff it
    // falls below the threshold and is carved out of the field as though it were
    // outside the camera's view, so every dark lesion was punching a hole in its
    // own analysis region and could never be found.
    const fitThr = Math.max(14, hi*0.20);
    const memberThr = Math.max(6, hi*0.06);

    let minX = w, maxX = -1, minY = h, maxY = -1;
    for (let y=0; y<h; y++){
      for (let x=0; x<w; x++){
        if (lum[y*w+x] <= fitThr) continue;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    const m = new Uint8Array(n);
    if (maxX < 0) return m;

    const cx = (minX+maxX)/2, cy = (minY+maxY)/2;
    const R = Math.max(maxX-minX, maxY-minY)/2;
    // Generous: trimming a tenth of the radius was measured to discard genuine
    // peripheral lesions. Large rim artifacts are handled during classification
    // by a rule keyed on size as well as position.
    const rIn = R*(1 - Math.max(0.02, erodeFrac*0.9));
    const rIn2 = rIn*rIn;

    for (let y=0; y<h; y++){
      for (let x=0; x<w; x++){
        const i = y*w+x;
        const dx = x-cx, dy = y-cy;
        if (dx*dx + dy*dy > rIn2) continue;
        if (lum[i] <= memberThr) continue;   // true black surround only
        m[i] = 1;
      }
    }
    return erodeMask(m, w, h, 2);
  }

  // Background estimate that only averages retina. A plain box blur near the
  // field-of-view rim mixes in the black surround, which drags the local
  // background down and makes ordinary edge retina look brighter than its
  // neighbourhood. That is what produced a ring of spurious bright findings
  // around the border. Dividing by the blurred mask renormalises each window to
  // the pixels that actually carry image data.
  function maskedBlur(src, mask, w, h, radius){
    const n = src.length;
    const num = new Float32Array(n), den = new Float32Array(n);
    for (let i=0;i<n;i++){ num[i] = mask[i] ? src[i] : 0; den[i] = mask[i] ? 1 : 0; }
    const bn = boxBlur(num, w, h, radius);
    const bd = boxBlur(den, w, h, radius);
    const out = new Float32Array(n);
    for (let i=0;i<n;i++) out[i] = bd[i] > 1e-3 ? bn[i]/bd[i] : src[i];
    return out;
  }

  function normalizeInside(arr, inside, n){
    let lo = Infinity, hi = -Infinity;
    for (let i=0;i<n;i++){
      if (!inside[i]) continue;
      if (arr[i] < lo) lo = arr[i];
      if (arr[i] > hi) hi = arr[i];
    }
    const range = (hi-lo) || 1;
    const out = new Float32Array(n);
    for (let i=0;i<n;i++) out[i] = inside[i] ? (arr[i]-lo)/range : 0;
    return out;
  }

  // Grow the connected region containing (sx,sy) over everything at or above the
  // cut, bounded in both radius and area. Returns null the moment either bound is
  // passed: at that point the fill has left the structure it was asked to measure
  // and is running into open retina, and no number it could return would mean
  // anything. A refusal here leaves the caller's earlier estimate standing.
  function floodBrightRegion(blur, inside, w, h, sx, sy, cut, maxR, maxArea){
    const n = w*h;
    const start = sy*w + sx;
    if (!inside[start] || blur[start] < cut) return null;
    const mask = new Uint8Array(n);
    const stack = new Int32Array(n);
    let sp = 0;
    stack[sp++] = start; mask[start] = 1;
    let area = 0, sumX = 0, sumY = 0;
    const maxR2 = maxR*maxR;
    function push(nb, nx, ny){
      if (mask[nb] || !inside[nb] || blur[nb] < cut) return;
      const dx = nx-sx, dy = ny-sy;
      if (dx*dx + dy*dy > maxR2) return;
      mask[nb] = 1; stack[sp++] = nb;
    }
    while (sp > 0){
      const idx = stack[--sp];
      const y = (idx/w)|0, x = idx - y*w;
      area++; sumX += x; sumY += y;
      if (area > maxArea) return null;
      if (x > 0)   push(idx-1, x-1, y);
      if (x < w-1) push(idx+1, x+1, y);
      if (y > 0)   push(idx-w, x, y-1);
      if (y < h-1) push(idx+w, x, y+1);
    }
    return { mask, area, cx: sumX/area, cy: sumY/area };
  }

  // Median of a sampled set, used for the two brightness references below. Both
  // are read as medians rather than means because both regions are crossed by
  // things that do not belong to them — vessels leaving the disc, the odd lesion
  // in the surrounding ring — and a mean would follow them.
  function medianOf(values){
    if (!values.length) return null;
    values.sort((a,b) => a-b);
    return values[values.length >> 1];
  }

  // Optic disc. Brightness alone picks the wrong region whenever the photo has a
  // specular highlight or a blown-out patch, so brightness is combined with local
  // contrast: the disc carries the vessel trunk and a sharp rim and is therefore
  // bright AND textured, while flare is bright and smooth. The centre is then the
  // centroid of the bright plateau rather than one argmax pixel, and the radius
  // comes from that plateau's area instead of being assumed.
  function estimateOpticDisc(canvas){
    const { w, h, n, lum } = imageChannels(canvas);
    const inside = retinaFieldMask(lum, w, h, 0.03);

    const r = Math.max(5, Math.round(Math.min(w,h)*0.045));
    const blur = maskedBlur(lum, inside, w, h, r);

    const sq = new Float32Array(n);
    for (let i=0;i<n;i++) sq[i] = lum[i]*lum[i];
    const blurSq = maskedBlur(sq, inside, w, h, r);
    const sd = new Float32Array(n);
    for (let i=0;i<n;i++) sd[i] = Math.sqrt(Math.max(0, blurSq[i] - blur[i]*blur[i]));

    const nb = normalizeInside(blur, inside, n);
    const ns = normalizeInside(sd, inside, n);

    let best = -Infinity, bestIdx = -1;
    for (let i=0;i<n;i++){
      if (!inside[i]) continue;
      const score = nb[i]*0.65 + ns[i]*0.35;
      if (score > best){ best = score; bestIdx = i; }
    }
    if (bestIdx < 0) throw new Error("no usable retinal area for disc estimation");

    let mean = 0, cnt = 0;
    for (let i=0;i<n;i++) if (inside[i]){ mean += blur[i]; cnt++; }
    mean = cnt ? mean/cnt : 0;
    const peak = blur[bestIdx];
    const cut = peak - (peak-mean)*0.35;

    const seedY = (bestIdx/w)|0, seedX = bestIdx - seedY*w;
    const box = Math.round(Math.min(w,h)*0.16);
    let sumX = 0, sumY = 0, area = 0;
    for (let y=Math.max(0,seedY-box); y<Math.min(h,seedY+box); y++){
      for (let x=Math.max(0,seedX-box); x<Math.min(w,seedX+box); x++){
        const i = y*w+x;
        if (!inside[i] || blur[i] < cut) continue;
        sumX += x; sumY += y; area++;
      }
    }

    const rMin = Math.min(w,h)*0.035, rMax = Math.min(w,h)*0.11;
    let discX = area ? sumX/area : seedX;
    let discY = area ? sumY/area : seedY;
    let radius = clamp(Math.sqrt(Math.max(area,1)/Math.PI), rMin, rMax);

    // ---- Measured extent, replacing the plateau area above wherever it can.
    //
    // The plateau is a cut at a fixed height above the mean of the whole retina,
    // and the whole retina is not what the disc's edge is defined against. Two
    // things go wrong with that. The reference is global, so any change in
    // average brightness elsewhere in the frame moves the disc's boundary; and
    // the height is fixed at 65% of the way to the peak, which on a soft-edged
    // structure is a contour inside the rim rather than the rim itself. Both push
    // the same way — the measured radius comes out a few per cent small, and the
    // exclusion circle built on it stops short of the disc's own margin.
    //
    // Growing the region instead fixes both. The level is referred to the retina
    // immediately around the disc, so it is local; and it is placed half way
    // between that background and the disc, which is where the half-maximum
    // contour of a blurred edge sits — the edge itself, rather than a contour
    // some fixed distance inside it.
    const cx0 = clampInt(Math.round(discX), 0, w-1);
    const cy0 = clampInt(Math.round(discY), 0, h-1);
    const r0 = radius;

    // Peripapillary background: retina in an annulus outside the disc, close
    // enough to share its illumination. The inner edge is held clear of the blur
    // radius as well as of the disc, because the same blur that makes the disc a
    // plateau also spreads it outward, and an annulus inside that spill would
    // read the disc back as its own background and shrink the fill to nothing.
    const ringInner = Math.max(r0*2.5, r0 + r*1.5);
    const ringOuter = Math.max(ringInner*1.6, r0*4.0);
    const ring = [];
    {
      const i2 = ringInner*ringInner, o2 = ringOuter*ringOuter;
      const ya = Math.max(0, Math.round(cy0-ringOuter)), yb = Math.min(h-1, Math.round(cy0+ringOuter));
      const xa = Math.max(0, Math.round(cx0-ringOuter)), xb = Math.min(w-1, Math.round(cx0+ringOuter));
      for (let y=ya; y<=yb; y++){
        const dy = y-cy0;
        for (let x=xa; x<=xb; x++){
          const dx = x-cx0, d2 = dx*dx + dy*dy;
          if (d2 < i2 || d2 > o2) continue;
          const i = y*w+x;
          if (inside[i]) ring.push(blur[i]);
        }
      }
    }
    // Too few pixels means the disc sits near the edge of the aperture and no
    // annulus fits; the global mean is a worse reference but it is the only one
    // left, and it errs toward a smaller fill rather than a runaway one.
    const ppBg = (ring.length > 200 ? medianOf(ring) : mean);

    // The disc's own level, as a high quantile of its core rather than the single
    // brightest pixel in it, so one specular speck on the cup cannot set the
    // scale that both cuts are measured against.
    const core = [];
    {
      const rc = Math.max(2, r0*0.6), rc2 = rc*rc;
      for (let y=Math.max(0,Math.round(cy0-rc)); y<=Math.min(h-1,Math.round(cy0+rc)); y++){
        const dy = y-cy0;
        for (let x=Math.max(0,Math.round(cx0-rc)); x<=Math.min(w-1,Math.round(cx0+rc)); x++){
          const dx = x-cx0;
          if (dx*dx + dy*dy > rc2) continue;
          const i = y*w+x;
          if (inside[i]) core.push(blur[i]);
        }
      }
      core.sort((a,b) => a-b);
    }
    const discLevel = core.length
      ? core[Math.min(core.length-1, Math.floor(core.length*0.75))]
      : blur[bestIdx];

    let tissue = null, measuredExtent = false;
    const span = discLevel - ppBg;
    // A span of a grey level or less means the disc is not separable from the
    // retina around it on this frame. Anything grown from that is noise, so
    // nothing is grown and the plateau estimate is left to stand alone.
    if (span > 1){
      const maxR = r0*DISC_FILL_MAX_MULT;
      const maxArea = Math.PI*Math.pow(r0*2.8, 2);
      const edge = floodBrightRegion(blur, inside, w, h, cx0, cy0, ppBg + span*DISC_EDGE_LEVEL, maxR, maxArea);
      if (edge){
        radius = clamp(Math.sqrt(edge.area/Math.PI), rMin, rMax);
        discX = edge.cx; discY = edge.cy;
        tissue = edge.mask;
        measuredExtent = true;
      }
      // The halo takes in whatever the disc shades into: the rim it was grown
      // from, a scleral crescent beside it, the nerve-fibre reflex arcing off it.
      // It supersedes the edge mask when it can be grown, being the larger of the
      // two and the one the bright-candidate test actually wants.
      const halo = floodBrightRegion(blur, inside, w, h, cx0, cy0, ppBg + span*DISC_HALO_LEVEL, maxR, maxArea);
      if (halo) tissue = halo.mask;
    }

    return {
      x: discX,
      y: discY,
      radius,
      score: best,
      // The tissue the disc is actually made of, at the resolution of the canvas
      // it was measured on — both dimensions travel with it, because a mask
      // indexed against the wrong size would delete findings somewhere else in
      // the image without saying so. Null when the fill could not be trusted, in
      // which case the exclusion circle stands on its own as it did before.
      tissue, tissueW: w, tissueH: h,
      measuredExtent
    };
  }

  // Fovea. Pure geometry from the disc puts the marker in roughly the right area
  // but rarely on the fovea itself, so the position is measured instead: the fovea
  // is the darkest part of the central retina, being avascular and pigment-dense.
  // Blurring at disc scale removes vessels and individual lesions as competitors,
  // and the search is restricted to the annulus and direction where the fovea can
  // anatomically be. The marker is the centroid of the dark plateau, which centres
  // it in the macular depression rather than on a single dark pixel.
  function estimateFoveaMacula(canvas, disc){
    // The macula is a broad feature, so the search runs on a downscaled copy.
    // That makes the morphology cheap and stops single dark pixels from mattering.
    const small = resizeCanvasCopy(canvas, 256);
    const sf = small.width / canvas.width;
    const { w, h, n, lum } = imageChannels(small);
    const inside = retinaFieldMask(lum, w, h, 0.06);

    const dx = disc.x*sf, dy = disc.y*sf;
    const dd = Math.max(6, disc.radius*2*sf);
    const sign = ((canvas.width/2) - disc.x) >= 0 ? 1 : -1;

    // Remove the vessels outright before looking for the darkest region.
    const closed = maxFilter(lum, w, h, Math.max(2, Math.round(dd*0.16)));
    const vesselFree = minFilter(closed, w, h, Math.max(2, Math.round(dd*0.16)));

    // How much darker than the surrounding retina, at macular scale.
    const bg = maskedBlur(vesselFree, inside, w, h, Math.max(6, Math.round(dd*1.5)));
    const darkness = new Float32Array(n);
    let dMax = 0;
    for (let i=0;i<n;i++){
      if (!inside[i]) continue;
      const v = bg[i] - vesselFree[i];
      darkness[i] = v > 0 ? v : 0;
      if (darkness[i] > dMax) dMax = darkness[i];
    }
    if (dMax <= 0) dMax = 1;

    // Anatomical anchor: the fovea lies about 2.5 disc diameters temporal to the
    // disc and a little below its level. Scoring darkness alone let any dark
    // structure anywhere in the search window capture the marker; weighting it
    // by distance from this expected point keeps the estimate anchored where the
    // fovea has to be, while still letting real macular darkening move it.
    const ex = dx + sign*dd*2.5;
    const ey = dy + dd*0.3;
    const sigma = dd*0.8, twoSig2 = 2*sigma*sigma;
    const reach = Math.round(sigma*2);

    let best = -Infinity, bestIdx = -1;
    for (let y=Math.max(0, Math.round(ey-reach)); y<Math.min(h, Math.round(ey+reach)); y++){
      for (let x=Math.max(0, Math.round(ex-reach)); x<Math.min(w, Math.round(ex+reach)); x++){
        const i = y*w+x;
        if (!inside[i]) continue;
        const ddx = x-ex, ddy = y-ey;
        const prior = Math.exp(-(ddx*ddx + ddy*ddy)/twoSig2);
        // The constant keeps the prior in charge when there is no dark signal,
        // so a featureless macula degrades to the anatomical position instead of
        // snapping to noise.
        const score = (0.4 + 0.6*(darkness[i]/dMax))*prior;
        if (score > best){ best = score; bestIdx = i; }
      }
    }

    let fovea, evidence;
    if (bestIdx >= 0){
      // Score-weighted centroid, so the marker sits in the middle of the dark
      // area rather than on its darkest pixel.
      const by = (bestIdx/w)|0, bx = bestIdx - by*w;
      const box = Math.round(dd*0.5);
      let sx = 0, sy = 0, swt = 0;
      for (let y=Math.max(0,by-box); y<Math.min(h,by+box); y++){
        for (let x=Math.max(0,bx-box); x<Math.min(w,bx+box); x++){
          const i = y*w+x;
          if (!inside[i]) continue;
          const ddx = x-ex, ddy = y-ey;
          const wt = (0.4 + 0.6*(darkness[i]/dMax))*Math.exp(-(ddx*ddx+ddy*ddy)/twoSig2);
          if (wt < best*0.75) continue;
          sx += x*wt; sy += y*wt; swt += wt;
        }
      }
      const fx = swt ? sx/swt : bx, fy = swt ? sy/swt : by;
      fovea = { x: fx/sf, y: fy/sf };
      evidence = "measured — darkest macular region after vessel removal, anchored to the expected position relative to the disc";
    } else {
      fovea = { x: ex/sf, y: ey/sf };
      evidence = "geometry only — no usable retinal area in the expected zone, so the anatomical position is used unrefined";
    }

    const margin = disc.radius*1.2;
    fovea.x = clamp(fovea.x, margin, canvas.width-margin);
    fovea.y = clamp(fovea.y, margin, canvas.height-margin);

    // The disc is nasal to the fovea in either eye, so the side the disc sits on
    // is the nasal side, with no need to know OD from OS.
    const nasalSide = (fovea.x - disc.x) >= 0 ? "left" : "right";
    return { fovea, maculaRadius: disc.radius*2*1.1, dir:{x:sign, y:0}, nasalSide, evidence };
  }

  function quadrantLabel(px, py, disc, nasalSide){
    const vertical = py < disc.y ? "Superior" : "Inferior";
    const leftOfDisc = px < disc.x;
    const horizontal = (nasalSide === "left")
      ? (leftOfDisc ? "Nasal" : "Temporal")
      : (leftOfDisc ? "Temporal" : "Nasal");
    return `${vertical}-${horizontal}`;
  }

  // ---------------------------------------------------------------------
  // DEFECT REGION MARKING: connected components of the Grad-CAM above a
  // fraction of peak activation, via iterative flood fill (no recursion).
  // ---------------------------------------------------------------------
  //
  // Two things decide the threshold, and the second one is why an image with
  // nothing to find now produces no circles at all. Grad-CAM is scaled to its
  // own maximum, so some pixel is always 1.0 and a fixed fraction-of-peak cut
  // always returns a region — on a healthy retina, the tallest bump in a flat
  // map. Requiring the region to also stand clear of the retina's own typical
  // activation removes exactly that case and leaves a genuine hotspot, which
  // clears both cuts easily, untouched. `inside` restricts everything to the
  // retina, so the black surround can neither be marked nor set the scale.
  function findDefectBlobs(camArray, h, w, threshold, minAreaFrac, inside, stats){
    const total = h*w;
    const minArea = Math.max(20, Math.round(total*minAreaFrac));
    const visited = new Uint8Array(total);
    const stack = new Int32Array(total);
    const blobs = [];

    // A map that is warm everywhere localises nothing, whatever its peak.
    if (stats && stats.median >= CAM_FLAT_MEDIAN) return [];
    const noiseCut = stats ? stats.median + CAM_NOISE_K*stats.mad : 0;
    const cut = Math.max(threshold, noiseCut);
    const isIn = inside ? (i => inside[i] === 1) : (() => true);

    for (let start=0; start<total; start++){
      if (visited[start] || !isIn(start) || camArray[start] < cut) continue;
      let sp = 0;
      stack[sp++] = start;
      visited[start] = 1;
      let sumX = 0, sumY = 0, count = 0, peak = 0;

      while (sp > 0){
        const idx = stack[--sp];
        const y = (idx/w)|0, x = idx - y*w;
        sumX += x; sumY += y; count++;
        if (camArray[idx] > peak) peak = camArray[idx];

        // 4-connectivity, guarding against row wrap-around on the horizontal neighbours
        if (x > 0){
          const n = idx-1;
          if (!visited[n] && isIn(n) && camArray[n] >= cut){ visited[n]=1; stack[sp++]=n; }
        }
        if (x < w-1){
          const n = idx+1;
          if (!visited[n] && isIn(n) && camArray[n] >= cut){ visited[n]=1; stack[sp++]=n; }
        }
        if (y > 0){
          const n = idx-w;
          if (!visited[n] && isIn(n) && camArray[n] >= cut){ visited[n]=1; stack[sp++]=n; }
        }
        if (y < h-1){
          const n = idx+w;
          if (!visited[n] && isIn(n) && camArray[n] >= cut){ visited[n]=1; stack[sp++]=n; }
        }
      }

      if (count >= minArea){
        const cx = sumX/count, cy = sumY/count;
        // Second pass over the component would be needed for an exact enclosing
        // radius; the area-equivalent radius is a good enough marker size here.
        const radius = Math.sqrt(count/Math.PI)*1.25;
        blobs.push({ cx, cy, area: count, peak, radius });
      }
    }

    blobs.sort((a,b) => b.peak - a.peak || b.area - a.area);
    return blobs.slice(0, CAM_REGION_LIMIT);
  }

  // Turbo-like ramp. Ordered so lightness rises with the value, which keeps the
  // tones readable in the order they mean something; a plain rainbow does not.
  const CAM_STOPS = [
    [0.00,  48, 18, 59], [0.15,  60,100,200], [0.30,  30,170,220],
    [0.45,  40,210,150], [0.60, 150,225, 60], [0.75, 250,205, 40],
    [0.90, 240,110, 30], [1.00, 160, 20, 20]
  ];
  function camColour(t){
    if (t <= 0) return CAM_STOPS[0].slice(1);
    if (t >= 1) return CAM_STOPS[CAM_STOPS.length-1].slice(1);
    for (let i=1;i<CAM_STOPS.length;i++){
      if (t <= CAM_STOPS[i][0]){
        const a = CAM_STOPS[i-1], b = CAM_STOPS[i];
        const f = (t - a[0])/(b[0] - a[0]);
        return [a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f, a[3]+(b[3]-a[3])*f];
      }
    }
    return CAM_STOPS[CAM_STOPS.length-1].slice(1);
  }

  function renderGradCAMColour(sourceCanvas, camArray, targetCanvas, inside, feather){
    const w = sourceCanvas.width, h = sourceCanvas.height;
    targetCanvas.width = w; targetCanvas.height = h;
    const ctx = targetCanvas.getContext("2d");
    ctx.drawImage(sourceCanvas, 0, 0);
    const imgData = ctx.getImageData(0,0,w,h);
    const d = imgData.data;
    for (let i=0, px=0; px<w*h; i+=4, px++){
      // Outside the analysed field the map has no meaning, so it is left
      // unpainted rather than tinted with the bottom of the palette — the tint
      // reads as a measurement, and there is none there.
      const fw = feather ? feather[px] : (inside ? inside[px] : 1);
      if (fw <= 0.002) continue;
      const t = camArray[px];
      const c = camColour(t);
      // Opacity rises with the value, so weak areas keep showing the retina
      // instead of being flooded with the low end of the palette, and falls to
      // nothing across the edge of the analysed field.
      const a = (0.20 + 0.55*t) * fw;
      d[i]   = d[i]*(1-a)   + c[0]*a;
      d[i+1] = d[i+1]*(1-a) + c[1]*a;
      d[i+2] = d[i+2]*(1-a) + c[2]*a;
    }
    ctx.putImageData(imgData, 0, 0);
  }

  function renderGradCAMOverlay(sourceCanvas, camArray, camH, camW, targetCanvas, inside, feather){
    const w = sourceCanvas.width, h = sourceCanvas.height;
    targetCanvas.width = w; targetCanvas.height = h;
    const ctx = targetCanvas.getContext("2d");
    ctx.drawImage(sourceCanvas, 0, 0);
    const imgData = ctx.getImageData(0,0,w,h);
    const d = imgData.data;
    const base = 0.55;
    for (let y=0; y<h; y++){
      for (let x=0; x<w; x++){
        const idx = y*w + x;
        const fw = feather ? feather[idx] : (inside ? inside[idx] : 1);
        if (fw <= 0.002) continue;
        const alpha = base * fw;
        const a = camArray[idx];
        const overlayGray = a*255;
        const p = idx*4;
        d[p]   = d[p]*(1-alpha)   + overlayGray*alpha;
        d[p+1] = d[p+1]*(1-alpha) + overlayGray*alpha;
        d[p+2] = d[p+2]*(1-alpha) + overlayGray*alpha;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // ---------------------------------------------------------------------
  // CANVAS ANNOTATION HELPERS
  // Text gets a white halo so it stays readable over both the dark and light
  // parts of the image, without introducing any colour.
  // ---------------------------------------------------------------------
  function haloText(ctx, text, x, y, align, baseline, bold, fontPx){
    ctx.save();
    ctx.textAlign = align;
    ctx.textBaseline = baseline;
    ctx.font = `${bold ? "bold " : ""}${fontPx}px -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`;
    ctx.lineJoin = "round";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#fff";
    ctx.strokeText(text, x, y);
    ctx.fillStyle = "#000";
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  function strokeCircle(ctx, x, y, r, dash){
    ctx.save();
    ctx.setLineDash(dash || []);
    ctx.lineWidth = 4; ctx.strokeStyle = "#fff";
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2); ctx.stroke();
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#000";
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2); ctx.stroke();
    ctx.restore();
  }

  function strokeSquare(ctx, x, y, r, dash){
    ctx.save();
    ctx.setLineDash(dash || []);
    ctx.lineWidth = 4; ctx.strokeStyle = "#fff";
    ctx.strokeRect(x-r, y-r, r*2, r*2);
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#000";
    ctx.strokeRect(x-r, y-r, r*2, r*2);
    ctx.restore();
  }

  function drawLesionMarker(ctx, shape, x, y, r){
    if (shape === "square")          strokeSquare(ctx, x, y, r);
    else if (shape === "squareDash") strokeSquare(ctx, x, y, r, [3,3]);
    else if (shape === "ring2"){     strokeCircle(ctx, x, y, r); strokeCircle(ctx, x, y, Math.max(1.5, r*0.45)); }
    else                             strokeCircle(ctx, x, y, r);
  }

  function drawDefectMarkers(targetCanvas, blobs, anatomy){
    const ctx = targetCanvas.getContext("2d");
    const w = targetCanvas.width, h = targetCanvas.height;
    const fontPx = Math.max(11, Math.round(w*0.028));
    const findings = [];
    let discCount = 0;

    // The optic disc draws strong attention from almost any fundus classifier:
    // it is the brightest, most distinctive structure in the frame, and a network
    // uses it to orient itself. That attention is real and worth showing, but it
    // is not a finding, so disc regions are drawn dashed and labelled rather than
    // numbered among the results.
    function onDisc(b){
      if (!anatomy) return false;
      return Math.hypot(b.cx-anatomy.disc.x, b.cy-anatomy.disc.y) <= anatomy.disc.radius*1.2;
    }

    let n = 0;
    blobs.forEach(b => {
      const r = Math.max(9, b.radius);
      if (onDisc(b)){
        discCount++;
        strokeCircle(ctx, b.cx, b.cy, r, [4,3]);
        haloText(ctx, "disc", b.cx, clamp(b.cy - r - 3, fontPx, h-2), "center", "bottom", false, fontPx);
        return;
      }
      n++;
      strokeCircle(ctx, b.cx, b.cy, r);
      haloText(ctx, String(n), b.cx, clamp(b.cy - r - 3, fontPx, h-2), "center", "bottom", true, fontPx);
      findings.push({
        index: n,
        quadrant: anatomy ? quadrantLabel(b.cx, b.cy, anatomy.disc, anatomy.nasalSide) : "n/a",
        peak: b.peak,
        areaPct: (b.area/(w*h))*100,
        areaPx: b.area
      });
    });
    return { findings, discCount };
  }

  function drawAnatomyOverlay(sourceCanvas, anatomy, targetCanvas){
    const w = sourceCanvas.width, h = sourceCanvas.height;
    targetCanvas.width = w; targetCanvas.height = h;
    const ctx = targetCanvas.getContext("2d");
    ctx.drawImage(sourceCanvas, 0, 0);

    const disc = anatomy.disc, fovea = anatomy.fovea;
    const fontPx = Math.max(10, Math.round(w*0.024));
    const pad = Math.round(fontPx*0.5) + 2;

    // Quadrant axes through the optic disc
    ctx.save();
    ctx.setLineDash([6,4]);
    ctx.lineWidth = 3; ctx.strokeStyle = "#fff";
    ctx.beginPath();
    ctx.moveTo(0, disc.y); ctx.lineTo(w, disc.y);
    ctx.moveTo(disc.x, 0); ctx.lineTo(disc.x, h);
    ctx.stroke();
    ctx.lineWidth = 1; ctx.strokeStyle = "#000";
    ctx.beginPath();
    ctx.moveTo(0, disc.y); ctx.lineTo(w, disc.y);
    ctx.moveTo(disc.x, 0); ctx.lineTo(disc.x, h);
    ctx.stroke();
    ctx.restore();

    // Quadrant names in the four frame corners
    haloText(ctx, quadrantLabel(pad, pad, disc, anatomy.nasalSide), pad, pad, "left", "top", false, fontPx);
    haloText(ctx, quadrantLabel(w-pad, pad, disc, anatomy.nasalSide), w-pad, pad, "right", "top", false, fontPx);
    haloText(ctx, quadrantLabel(pad, h-pad, disc, anatomy.nasalSide), pad, h-pad, "left", "bottom", false, fontPx);
    haloText(ctx, quadrantLabel(w-pad, h-pad, disc, anatomy.nasalSide), w-pad, h-pad, "right", "bottom", false, fontPx);

    // Macula (dashed) and fovea cross
    strokeCircle(ctx, fovea.x, fovea.y, anatomy.maculaRadius, [3,3]);
    const s = Math.max(4, w*0.009);
    ctx.save();
    ctx.lineWidth = 3; ctx.strokeStyle = "#fff";
    ctx.beginPath();
    ctx.moveTo(fovea.x-s, fovea.y); ctx.lineTo(fovea.x+s, fovea.y);
    ctx.moveTo(fovea.x, fovea.y-s); ctx.lineTo(fovea.x, fovea.y+s);
    ctx.stroke();
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#000";
    ctx.beginPath();
    ctx.moveTo(fovea.x-s, fovea.y); ctx.lineTo(fovea.x+s, fovea.y);
    ctx.moveTo(fovea.x, fovea.y-s); ctx.lineTo(fovea.x, fovea.y+s);
    ctx.stroke();
    ctx.restore();

    // Optic disc
    strokeCircle(ctx, disc.x, disc.y, disc.radius);

    // Labels, kept inside the frame
    haloText(ctx, "Optic disc", disc.x, clamp(disc.y - disc.radius - 3, fontPx, h-2), "center", "bottom", true, fontPx);
    haloText(ctx, "Macula", fovea.x, clamp(fovea.y - anatomy.maculaRadius - 3, fontPx, h-2), "center", "bottom", true, fontPx);
    haloText(ctx, "Fovea", clamp(fovea.x + s + 3, 0, w), fovea.y, "left", "middle", false, fontPx);
  }

  function renderFindingsLegend(findings, discCount){
    const container = document.getElementById("findings-legend");
    const discNote = discCount
      ? `<p class="small">${discCount} further region${discCount===1?" was":"s were"} centred on the optic disc and ${discCount===1?"is":"are"} drawn dashed rather than numbered. Attention there is expected — the disc is the most distinctive structure in the frame and the network uses it to orient itself — so it is shown but not counted as a finding.</p>`
      : "";
    if (!findings || findings.length === 0){
      container.innerHTML = `<p class="small">No region of the gradient exceeded the marking threshold — the model's attention is spread out rather than concentrated in discrete spots, so nothing is circled above.</p>` + discNote;
      return;
    }
    let rows = findings.map(f => `
      <tr>
        <td>Region ${f.index}</td>
        <td>${f.quadrant} quadrant · peak activation ${(f.peak*100).toFixed(0)}% of maximum · ${f.areaPct.toFixed(1)}% of frame area</td>
      </tr>`).join("");
    container.innerHTML = `<table class="kv">${rows}</table>` + discNote;
  }

  function renderAnatomyTable(anatomy){
    const t = document.getElementById("anatomy-table");
    const nasalWord = anatomy.nasalSide === "left" ? "left" : "right";
    const temporalWord = anatomy.nasalSide === "left" ? "right" : "left";
    t.innerHTML = `
      <tr><td>Optic disc centre (estimated)</td><td>x ${Math.round(anatomy.disc.x)}, y ${Math.round(anatomy.disc.y)} px · radius ${Math.round(anatomy.disc.radius)} px<br><span class="small">${anatomy.disc.measuredExtent
        ? "grown outward from the disc's own brightness down to the half-way level between it and the retina immediately around it, so the radius follows the rim rather than the area of whatever plateau a global threshold happened to leave"
        : "from the area of the bright plateau — the extent could not be grown on this image, so this radius may undersize the disc"}</span></td></tr>
      <tr><td>Fovea centre (estimated)</td><td>x ${Math.round(anatomy.fovea.x)}, y ${Math.round(anatomy.fovea.y)} px<br><span class="small">${anatomy.evidence || "method not recorded"}</span></td></tr>
      <tr><td>Macula (estimated)</td><td>circle of radius ${Math.round(anatomy.maculaRadius)} px around the fovea</td></tr>
      <tr><td>Nasal side of frame</td><td>${nasalWord} of the disc axis (the disc is nasal to the fovea in either eye)</td></tr>
      <tr><td>Temporal side of frame</td><td>${temporalWord} of the disc axis (toward the macula)</td></tr>
    `;
  }

  // ---------------------------------------------------------------------
  // STEP 4: LESION CANDIDATE DETECTION
  //
  // Classical morphology, entirely independent of the CNN. The method is:
  //   1. Build a retina mask and erode it, so the field-of-view edge — a huge
  //      intensity step — cannot generate candidates.
  //   2. Estimate local background with a large box blur, then look at each
  //      pixel's deviation from it. This is what makes the detector insensitive
  //      to uneven illumination.
  //   3. Threshold those deviations at a multiple of their own standard
  //      deviation, so the sensitivity adapts to the image rather than being
  //      fixed for one camera.
  //   4. Group surviving pixels into connected components and keep or reject
  //      each one on measured shape and contrast.
  // ---------------------------------------------------------------------
  // Each type gets its own marker shape so every mark on the overlay is
  // identifiable on sight. Labelling only the largest few left most marks
  // anonymous, which made the overlay impossible to read.
  const LESION_TYPES = {
    MA:  { key:"MA",  label:"Microaneurysm candidate",        grey:255, shape:"circle",     shapeName:"circle" },
    HEM: { key:"HEM", label:"Dot/blot hemorrhage candidate",  grey:200, shape:"ring2",      shapeName:"double circle" },
    HE:  { key:"HE",  label:"Hard exudate candidate",         grey:150, shape:"square",     shapeName:"square" },
    CWS: { key:"CWS", label:"Cotton wool spot candidate",     grey:100, shape:"squareDash", shapeName:"dashed square" }
  };
  const LESION_ORDER = ["MA","HEM","HE","CWS"];

  // The "4" arm of the ICDR 4-2-1 rule for severe NPDR: more than twenty
  // intraretinal hemorrhages in each of the four quadrants.
  const ICDR_HEM_PER_QUADRANT = 20;
  const QUADRANT_NAMES = ["Superior-Nasal","Superior-Temporal","Inferior-Nasal","Inferior-Temporal"];

  function erodeMask(mask, w, h, radius){
    const f = new Float32Array(mask.length);
    for (let i=0;i<mask.length;i++) f[i] = mask[i];
    const b = boxBlur(f, w, h, radius);
    const out = new Uint8Array(mask.length);
    // A box-blurred binary mask equals 1 only where every pixel in the window is
    // 1, so this thresholding is an exact erosion by a square structuring element.
    for (let i=0;i<mask.length;i++) out[i] = b[i] >= 0.999 ? 1 : 0;
    return out;
  }

  // Connected components with the shape and contrast features the type rules need.
  function connectedComponents(binary, w, h, maxArea, contrastMap, gradientMap, seedMask){
    const n = w*h;
    const visited = new Uint8Array(n);
    const stack = new Int32Array(n);
    const comps = [];

    for (let s=0; s<n; s++){
      if (visited[s] || !binary[s]) continue;
      let sp = 0;
      stack[sp++] = s; visited[s] = 1;
      let count = 0, sumX = 0, sumY = 0;
      let minX = w, maxX = 0, minY = h, maxY = 0;
      let sumContrast = 0, sumGradient = 0;
      let maxContrast = 0, hasSeed = seedMask ? false : true;
      const pixels = [];

      while (sp > 0){
        const idx = stack[--sp];
        const y = (idx/w)|0, x = idx - y*w;
        count++; sumX += x; sumY += y;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        sumContrast += contrastMap[idx];
        sumGradient += gradientMap[idx];
        if (contrastMap[idx] > maxContrast) maxContrast = contrastMap[idx];
        if (seedMask && seedMask[idx]) hasSeed = true;
        // Stop accumulating pixels once the component is already too large to
        // keep; it will be discarded, and this bounds memory on vessel networks.
        if (pixels.length <= maxArea) pixels.push(idx);

        if (x > 0){        const nb = idx-1; if (binary[nb] && !visited[nb]){ visited[nb]=1; stack[sp++]=nb; } }
        if (x < w-1){      const nb = idx+1; if (binary[nb] && !visited[nb]){ visited[nb]=1; stack[sp++]=nb; } }
        if (y > 0){        const nb = idx-w; if (binary[nb] && !visited[nb]){ visited[nb]=1; stack[sp++]=nb; } }
        if (y < h-1){      const nb = idx+w; if (binary[nb] && !visited[nb]){ visited[nb]=1; stack[sp++]=nb; } }
      }

      const bw = maxX-minX+1, bh = maxY-minY+1;
      comps.push({
        pixels, area: count,
        cx: sumX/count, cy: sumY/count,
        bw, bh,
        fillRatio: count/(bw*bh),
        aspect: Math.max(bw,bh)/Math.max(1, Math.min(bw,bh)),
        meanContrast: sumContrast/count,
        meanGradient: sumGradient/count,
        maxContrast, hasSeed
      });
    }
    return comps;
  }

  // Robust scale estimate. Standard deviation is the wrong statistic here: the
  // vessel tree is a large population of strong dark deviations and inflates it,
  // which pushes the threshold up and hides real lesions. Median absolute
  // deviation ignores that minority and tracks the actual noise floor. Computed
  // through histograms so it stays a linear pass.
  function robustSigma(dev, inside, n){
    const BINS = 320, LO = -80, HI = 80, sc = BINS/(HI-LO);
    // A noise scale does not need every pixel; a few hundred thousand samples
    // fix it as precisely as the full image does.
    const step = n > 300000 ? 2 : 1;
    const hist = new Int32Array(BINS);
    let count = 0;
    for (let i=0;i<n;i+=step){
      if (!inside[i]) continue;
      let v = dev[i];
      if (v < LO) v = LO; else if (v > 79.999) v = 79.999;
      hist[((v-LO)*sc)|0]++;
      count++;
    }
    if (!count) return 1;
    const half = count/2;
    let cum = 0, median = 0;
    for (let b=0;b<BINS;b++){
      cum += hist[b];
      if (cum >= half){ median = LO + (b+0.5)/sc; break; }
    }
    const ABINS = 320, AHI = 80, asc = ABINS/AHI;
    const ahist = new Int32Array(ABINS);
    for (let i=0;i<n;i+=step){
      if (!inside[i]) continue;
      let a = dev[i] - median;
      if (a < 0) a = -a;
      if (a > 79.999) a = 79.999;
      ahist[(a*asc)|0]++;
    }
    let acum = 0, mad = AHI;
    for (let b=0;b<ABINS;b++){
      acum += ahist[b];
      if (acum >= half){ mad = (b+0.5)/asc; break; }
    }
    return Math.max(0.5, 1.4826*mad);
  }

  // Grayscale max/min filters over a square window, used by the fovea search.
  // A square element is deliberately NOT used for lesion work: it removes
  // everything smaller than itself, and a microaneurysm is smaller than a vessel
  // is wide, so it cannot tell the two apart.
  function maxFilter(src, w, h, r){
    const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
    for (let y=0;y<h;y++){
      const row = y*w;
      for (let x=0;x<w;x++){
        let m = -Infinity;
        const x0 = Math.max(0,x-r), x1 = Math.min(w-1,x+r);
        for (let k=x0;k<=x1;k++){ const v = src[row+k]; if (v>m) m = v; }
        tmp[row+x] = m;
      }
    }
    for (let x=0;x<w;x++){
      for (let y=0;y<h;y++){
        let m = -Infinity;
        const y0 = Math.max(0,y-r), y1 = Math.min(h-1,y+r);
        for (let k=y0;k<=y1;k++){ const v = tmp[k*w+x]; if (v>m) m = v; }
        out[y*w+x] = m;
      }
    }
    return out;
  }

  function minFilter(src, w, h, r){
    const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
    for (let y=0;y<h;y++){
      const row = y*w;
      for (let x=0;x<w;x++){
        let m = Infinity;
        const x0 = Math.max(0,x-r), x1 = Math.min(w-1,x+r);
        for (let k=x0;k<=x1;k++){ const v = src[row+k]; if (v<m) m = v; }
        tmp[row+x] = m;
      }
    }
    for (let x=0;x<w;x++){
      for (let y=0;y<h;y++){
        let m = Infinity;
        const y0 = Math.max(0,y-r), y1 = Math.min(h-1,y+r);
        for (let k=y0;k<=y1;k++){ const v = tmp[k*w+x]; if (v<m) m = v; }
        out[y*w+x] = m;
      }
    }
    return out;
  }

  function directionalMorph(src, w, h, radius, mode, orientCount){
    const n = w*h;
    const maxLen = Math.max(w, h) + 2;
    const idx = new Int32Array(maxLen);
    const dq  = new Int32Array(maxLen);
    const gathered = new Float32Array(maxLen);
    const stage1 = new Float32Array(maxLen);
    const stage2 = new Float32Array(maxLen);
    const accMin = new Float32Array(n).fill(Infinity);
    const accMax = new Float32Array(n).fill(-Infinity);
    const ORIENT = orientCount || LINEAR_ORIENTATIONS;
    const closing = (mode === "close");

    // Running max/min along a gathered line, with a monotonic deque so cost does
    // not grow with element length.
    function run1D(sArr, dArr, len, r, isMax){
      let head = 0, tail = 0, next = 0;
      for (let i=0; i<len; i++){
        const hi = Math.min(len-1, i+r);
        while (next <= hi){
          const v = sArr[next];
          while (tail > head){
            const back = sArr[dq[tail-1]];
            if (isMax ? (back <= v) : (back >= v)) tail--; else break;
          }
          dq[tail++] = next++;
        }
        const lo = i - r;
        while (dq[head] < lo) head++;
        dArr[i] = sArr[dq[head]];
      }
    }

    // A closing along a line depends only on that line, so both stages run while
    // the line is gathered. Building the index list once per orientation instead
    // of once per stage roughly halves the work.
    function processLine(L){
      for (let i=0;i<L;i++) gathered[i] = src[idx[i]];
      run1D(gathered, stage1, L, radius, closing);    // close dilates first
      run1D(stage1, stage2, L, radius, !closing);     // then erodes
      for (let i=0;i<L;i++){
        const v = stage2[i], pos = idx[i];
        if (v < accMin[pos]) accMin[pos] = v;
        if (v > accMax[pos]) accMax[pos] = v;
      }
    }

    // Discrete lines at an arbitrary angle. Restricting these to the axes and
    // diagonals leaves vessels running at intermediate angles unprotected: no
    // element lies along them, every orientation cuts across, and the closing
    // fills them exactly as it fills a lesion. Each pixel belongs to exactly one
    // line of a given family, so the image is covered exactly once. The in-bounds
    // span is solved for rather than found by walking, because most lines in an
    // oblique family barely cross the image.
    function forEachLine(theta){
      const cos = Math.cos(theta), sin = Math.sin(theta);
      if (Math.abs(cos) >= Math.abs(sin)){
        const slope = sin/cos;
        const span = Math.abs(Math.round((w-1)*slope)) + 2;
        for (let y0 = -span; y0 <= h-1+span; y0++){
          let xLo = 0, xHi = w-1;
          if (slope > 1e-9){
            xLo = Math.max(0, Math.floor((-0.5 - y0)/slope));
            xHi = Math.min(w-1, Math.ceil((h-0.5 - y0)/slope));
          } else if (slope < -1e-9){
            xLo = Math.max(0, Math.floor((h-0.5 - y0)/slope));
            xHi = Math.min(w-1, Math.ceil((-0.5 - y0)/slope));
          } else if (y0 < 0 || y0 >= h){
            continue;
          }
          let L = 0;
          for (let x=xLo; x<=xHi; x++){
            const y = y0 + Math.round(x*slope);
            if (y >= 0 && y < h) idx[L++] = y*w + x;
            else if (L) break;
          }
          if (L > 0) processLine(L);
        }
      } else {
        const slope = cos/sin;
        const span = Math.abs(Math.round((h-1)*slope)) + 2;
        for (let x0 = -span; x0 <= w-1+span; x0++){
          let yLo = 0, yHi = h-1;
          if (slope > 1e-9){
            yLo = Math.max(0, Math.floor((-0.5 - x0)/slope));
            yHi = Math.min(h-1, Math.ceil((w-0.5 - x0)/slope));
          } else if (slope < -1e-9){
            yLo = Math.max(0, Math.floor((w-0.5 - x0)/slope));
            yHi = Math.min(h-1, Math.ceil((-0.5 - x0)/slope));
          } else if (x0 < 0 || x0 >= w){
            continue;
          }
          let L = 0;
          for (let y=yLo; y<=yHi; y++){
            const x = x0 + Math.round(y*slope);
            if (x >= 0 && x < w) idx[L++] = y*w + x;
            else if (L) break;
          }
          if (L > 0) processLine(L);
        }
      }
    }

    for (let o=0; o<ORIENT; o++) forEachLine(Math.PI*o/ORIENT);
    for (let i=0;i<n;i++){
      if (accMin[i] === Infinity){ accMin[i] = src[i]; accMax[i] = src[i]; }
    }
    // min over orientations isolates structures every direction filled, i.e.
    // round ones; max over orientations catches anything at least one direction
    // filled, i.e. everything thin, vessels included.
    return { min: accMin, max: accMax };
  }

  // Upper-quantile anchor for a one-sided map. robustSigma is the right tool for
  // the symmetric deviation maps, where the median really is the background and
  // the MAD really is the noise. It is the wrong tool for a top-hat, which is
  // non-negative and piles most of its mass on zero: there the MAD describes the
  // zero atom and ignores the tail entirely. This reads the tail directly.
  function tailQuantile(map, mask, n, q){
    const BINS = 2048, HI = 128, sc = BINS/HI;
    const step = n > 300000 ? 2 : 1;
    const hist = new Int32Array(BINS);
    let count = 0;
    for (let i=0;i<n;i+=step){
      if (!mask[i]) continue;
      let v = map[i];
      if (v < 0) v = 0; else if (v > HI-0.001) v = HI-0.001;
      hist[(v*sc)|0]++;
      count++;
    }
    if (!count) return 1;
    const target = count*q;
    let cum = 0;
    for (let b=0;b<BINS;b++){
      cum += hist[b];
      if (cum >= target) return (b+0.5)/sc;
    }
    return HI;
  }

  // Flood a strong mask outward through a permissive one. Both masks are already
  // gated on the same evidence, so this only ever connects what the permissive
  // threshold already believed; what it adds is the requirement that a weak pixel
  // be attached to a confident one, which is exactly what distinguishes the faint
  // continuation of a real vessel from an isolated patch of texture at the same
  // amplitude. Eight-connected, because a vessel crossing the pixel grid at an
  // angle is a staircase and four-connectivity breaks it into beads.
  function hysteresisMask(strong, weak, w, h){
    const n = w*h;
    const out = new Uint8Array(n);
    const stack = new Int32Array(n);
    let sp = 0;
    for (let i=0;i<n;i++) if (strong[i]){ out[i] = 1; stack[sp++] = i; }
    while (sp > 0){
      const idx = stack[--sp];
      const y = (idx/w)|0, x = idx - y*w;
      for (let dy=-1; dy<=1; dy++){
        const ny = y+dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx=-1; dx<=1; dx++){
          const nx = x+dx;
          if (nx < 0 || nx >= w) continue;
          const nb = ny*w + nx;
          if (!out[nb] && weak[nb]){ out[nb] = 1; stack[sp++] = nb; }
        }
      }
    }
    return out;
  }

  function roundStructureSeeds(greenS, lumS, inside, w, h){
    const n = w*h;
    const seL = Math.max(3, Math.round(Math.min(w,h)*LINEAR_SE_FRAC));
    const closed = directionalMorph(greenS, w, h, seL, "close");
    const opened = directionalMorph(lumS, w, h, seL, "open");

    // The same two top-hats at successively longer elements. Everything bounded
    // has already been filled at its own scale and does not respond any harder
    // at the next rung; everything that continues past the element does. The
    // difference between consecutive rungs is the saturation test.
    const satSE = [seL];
    const satDark = [], satBright = [];
    for (let k=1; k<SATURATION_LADDER; k++){
      satSE.push(Math.max(satSE[k-1]+1, Math.round(seL*Math.pow(SATURATION_SE_MULT, k))));
    }

    const roundDark = new Float32Array(n);
    const roundBright = new Float32Array(n);
    // Spread of the closing across orientations: near zero where every direction
    // filled the structure equally (round), large where one direction filled
    // nothing and the rest filled completely (elongated).
    const anisoDark = new Float32Array(n);
    for (let i=0;i<n;i++){
      if (!inside[i]) continue;
      const rd = closed.min[i] - greenS[i];      // filled by every direction: round
      const rb = lumS[i] - opened.max[i];
      roundDark[i] = rd > 0 ? rd : 0;
      roundBright[i] = rb > 0 ? rb : 0;
      const an = closed.max[i] - closed.min[i];
      anisoDark[i] = an > 0 ? an : 0;
    }
    satDark.push(roundDark); satBright.push(roundBright);
    for (let k=1; k<satSE.length; k++){
      const cB = directionalMorph(greenS, w, h, satSE[k], "close", SATURATION_ORIENTATIONS);
      const oB = directionalMorph(lumS, w, h, satSE[k], "open", SATURATION_ORIENTATIONS);
      const dk = new Float32Array(n), br = new Float32Array(n);
      for (let i=0;i<n;i++){
        if (!inside[i]) continue;
        const a = cB.min[i] - greenS[i];
        const b = lumS[i] - oB.max[i];
        dk[i] = a > 0 ? a : 0;
        br[i] = b > 0 ? b : 0;
      }
      satDark.push(dk); satBright.push(br);
    }

    const tDark = Math.max(SEED_FLOOR_DARK, tailQuantile(roundDark, inside, n, SEED_TAIL_Q));
    const tBright = Math.max(SEED_FLOOR_BRIGHT, tailQuantile(roundBright, inside, n, SEED_TAIL_Q));

    const darkSeed = new Uint8Array(n), brightSeed = new Uint8Array(n);
    for (let i=0;i<n;i++){
      if (!inside[i]) continue;
      if (roundDark[i] > tDark) darkSeed[i] = 1;
      if (roundBright[i] > tBright) brightSeed[i] = 1;
    }

    // Vessel map: elongated where the closing's spread across orientations
    // dominates its floor. The comparison is between the two responses rather
    // than against a fixed level, so the same rule holds for a thin peripheral
    // capillary and a wide vein without either being tuned for. Found at a strict
    // level and followed outward at a permissive one, so a vessel is traced along
    // its length instead of appearing only where it happens to be darkest.
    const tVesselSeed = Math.max(3, tailQuantile(anisoDark, inside, n, VESSEL_SEED_Q));
    const tVesselGrow = Math.max(2, tailQuantile(anisoDark, inside, n, VESSEL_GROW_Q));
    const vSeed = new Uint8Array(n), vGrow = new Uint8Array(n);
    for (let i=0;i<n;i++){
      if (!inside[i]) continue;
      // Elongated: the spread outruns what every direction filled in common. A
      // lesion fails this however dark it is, which is what stops the vessel map
      // from swallowing the findings it exists to protect.
      if (anisoDark[i] <= VESSEL_ELONGATION*roundDark[i]) continue;
      if (anisoDark[i] > tVesselSeed) vSeed[i] = 1;
      if (anisoDark[i] > tVesselGrow) vGrow[i] = 1;
    }
    const vesselMask = hysteresisMask(vSeed, vGrow, w, h);

    // Seeding stays on the shortest element, and the reason is worth recording
    // because the obvious improvement was measured and rejected.
    //
    // A closing only fills a structure narrower than the element, so a blot
    // hemorrhage wider than the element barely responds: on a planted hemorrhage
    // of radius 20 the mean top-hat at the nominal element is 0.4 grey levels and
    // not one of its 1257 pixels seeds, while at twice and four times that length
    // it answers at 25.4 and 34.7. Since a seed is mandatory, such a lesion can
    // never be reported however obvious it is. That is the real cause of the
    // large-hemorrhage blindness, and it is a property of the seeding scale, not
    // of the vessel-rejection rule the notes previously blamed.
    //
    // Seeding from the longer rungs as well does fix the seeding, and does not
    // fix the detection: the newly seeded regions are then rejected at the
    // contrast test instead, because a long element also responds to the macula
    // and to illumination falloff, which arrive together with the hemorrhages.
    // Measured over three retinas it raised false marks on healthy images from 19
    // to 29 while recovering one large hemorrhage out of six. The gap is real but
    // it lives further down, in how extent and contrast are measured, and adding
    // scales to the seeder only moves the failure rather than removing it.

    return { darkSeed, brightSeed, vesselMask, seL, satSE, satDark, satBright,
             roundDark, roundBright, tDark, tBright };
  }

  // ---------------------------------------------------------------------
  // VESSEL-JUNCTION TEST
  //
  // The directional top-hat that finds lesions has one systematic blind spot,
  // and it is the whole reason a healthy retina came back covered in
  // microaneurysm marks. The test asks whether a structure is filled in by a
  // linear closing in *every* direction; a vessel is not, because it survives
  // the element that lies along it. But that argument only holds where a vessel
  // is locally straight and alone. Where two vessels cross, where one branches,
  // and at the apex of a tight bend, there is no direction in which the
  // structure is straight — so every orientation fills it, it answers the
  // roundness test exactly as a lesion does, and it is seeded. Those three
  // configurations are common, they lie all over the vascular arcades, and they
  // are precisely the "veins" being reported as lesions.
  //
  // No refinement of the top-hat itself can separate them, because at the
  // junction the two really do look the same. What differs is the surroundings:
  // a lesion is an isolated blob with normal retina around it, while a junction
  // has vessel running out of it in three or four directions. So the region is
  // judged by what leaves it. A ring is sampled just outside the candidate and
  // divided into angular sectors:
  //
  //   0 or 1 vessel arm   free-standing lesion, or one sitting against a vessel
  //   2 opposite arms     a vessel passing straight through — a microaneurysm or
  //                       dot hemorrhage on a vessel looks like this, and those
  //                       are real, so this case is KEPT
  //   2 arms at an angle  a bend: the vessel turns here, and the turn is what was
  //                       detected
  //   3 or more arms      a crossing or a bifurcation
  //
  // Keeping the antipodal case is what stops this from throwing away genuine
  // lesions on vessels, which is the failure mode that matters most: a
  // microaneurysm beside a venule is a real finding and by far the commonest
  // early sign.
  // How far a two-armed candidate's arms may fall short of opposite before the
  // structure is read as a bend rather than a vessel running through. Deliberately
  // wide. A vessel curving gently past a small hemorrhage does not leave in
  // exactly opposite directions, and the cost of the two errors is not
  // symmetric: dropping a real lesion is worse than keeping a kink, so only a
  // pronounced turn is called a bend.
  const JUNCTION_ANTIPODAL_TOL_DEG = 45;
  // A candidate lying along the course of a vessel is not rejected — a
  // microaneurysm beside a venule is real, and by far the commonest early sign,
  // so deleting this class would cost more than it saves. It is held to a
  // stricter contrast bar instead, and reported as vessel-associated so a reader
  // can see which marks sit on the vasculature. The factor is deliberately
  // modest: enough to drop a wide spot in a vein that is only as dark as the
  // vein, not enough to drop a focal lesion on one.
  const ON_VESSEL_CONTRAST_FACTOR = 1.35;
  // A candidate with vessel persisting outward in this much of every direction
  // is inside the vascular tree, not a discrete lesion within it. Because the
  // measure is now persistence rather than mere presence, a lesion's collar
  // cannot reach this, and the threshold can stay strict.
  const JUNCTION_RING_SATURATION = 0.70;

  // Arms are counted on each ring separately and the busiest ring decides.
  // Unioning the rings first was measured to lose the commonest case of all: two
  // vessels branching at an acute angle are still touching a ring drawn close in,
  // so they merge into one wide arm and the bifurcation reads as a lone vessel.
  // They separate further out. Saturation is judged on the innermost ring
  // instead, because "buried in vasculature" is a statement about the
  // candidate's immediate surroundings, and a wide enough ring eventually hits
  // vessels on any retina.
  // Arms are found by walking outward, not by sampling a ring.
  //
  // A ring was tried first and failed for a reason worth recording, because it
  // looks like the obvious construction. The vessel map is built from pixels
  // that are thin in some direction but not round in all of them — and at the
  // *edge* of a round lesion that is exactly what the pixels are: the roundness
  // response has fallen away while the thinness response has not. So every
  // lesion carries a thin collar of false vessel a few pixels wide. A ring drawn
  // close enough to a microaneurysm to describe its surroundings lands squarely
  // in that collar, reads vessel in every direction, and the test throws away the
  // very lesions it exists to protect. Measured on synthetic retinas, the ring
  // version rejected three planted lesions and no vessels at all.
  //
  // Walking outward separates the two cleanly. A collar is a few pixels deep and
  // then stops; a vessel arm continues for as far as the walk goes. Requiring the
  // vessel to *persist* along a ray, rather than merely to be present at one
  // radius, is what tells a lesion's own edge from a vessel leaving a junction.
  const ARM_DIRECTIONS = 24;
  // How much of the walk must be on vessel before the direction counts as an arm.
  const ARM_PERSISTENCE = 0.55;
  // The walk starts clear of the candidate's own collar and runs about a vessel
  // element's length — far enough to outlast a collar, short enough that a
  // curving vessel has not yet left the ray.
  const ARM_COLLAR_PAD = 3;

  function vesselArms(cx, cy, area, vesselZone, w, h, armLength){
    const r0 = Math.sqrt(Math.max(area,1)/Math.PI);
    const inner = r0 + ARM_COLLAR_PAD;
    const outer = inner + Math.max(8, armLength);
    const steps = Math.max(6, Math.round(outer - inner));

    const hit = new Uint8Array(ARM_DIRECTIONS);
    for (let d=0; d<ARM_DIRECTIONS; d++){
      const th = 2*Math.PI*d/ARM_DIRECTIONS;
      const c = Math.cos(th), sn = Math.sin(th);
      // A perpendicular tolerance of one pixel keeps the ray on a vessel that
      // leans slightly, without letting it wander onto a neighbouring one.
      const px = -sn, py = c;
      let on = 0, seen = 0;
      for (let k=0; k<=steps; k++){
        const r = inner + (outer-inner)*k/steps;
        let any = 0;
        for (let o=-1; o<=1; o++){
          const x = Math.round(cx + r*c + o*px);
          const y = Math.round(cy + r*sn + o*py);
          if (x < 0 || y < 0 || x >= w || y >= h) continue;
          if (vesselZone[y*w + x]){ any = 1; break; }
        }
        seen++; on += any;
      }
      if (seen && on/seen >= ARM_PERSISTENCE) hit[d] = 1;
    }

    let hits = 0;
    for (let d=0; d<ARM_DIRECTIONS; d++) hits += hit[d];
    if (hits === 0) return { arms: 0, coverage: 0, angles: [] };
    if (hits === ARM_DIRECTIONS) return { arms: 1, coverage: 1, angles: [0] };

    // Contiguous runs of arm directions, counted around the circle: one vessel
    // several directions wide is one arm, not several.
    let start = 0;
    while (hit[start]) start++;                  // begin at a gap, so runs do not wrap
    const angles = [];
    let i = 0;
    while (i < ARM_DIRECTIONS){
      if (!hit[(start + i) % ARM_DIRECTIONS]){ i++; continue; }
      let len = 0, sum = 0;
      while (len < ARM_DIRECTIONS){
        const j = (start + i + len) % ARM_DIRECTIONS;
        if (!hit[j]) break;
        sum += (start + i + len);
        len++;
      }
      angles.push(((sum/len) % ARM_DIRECTIONS) * 360/ARM_DIRECTIONS);
      i += len;
    }
    return { arms: angles.length, coverage: hits/ARM_DIRECTIONS, angles };
  }

  function isVesselJunction(ctx){
    if (ctx.coverage >= JUNCTION_RING_SATURATION) return true;
    if (ctx.arms >= 3) return true;
    if (ctx.arms === 2){
      // Opposite arms are one vessel passing through, which a real lesion may
      // legitimately sit on. Anything else is the vessel changing direction.
      return !isThroughVessel(ctx);
    }
    return false;
  }

  // Two arms leaving in roughly opposite directions: a vessel runs through this
  // region rather than ending, branching or turning in it.
  function isThroughVessel(ctx){
    if (ctx.arms !== 2) return false;
    let d = Math.abs(ctx.angles[0] - ctx.angles[1]);
    if (d > 180) d = 360 - d;
    return Math.abs(d - 180) <= JUNCTION_ANTIPODAL_TOL_DEG;
  }

  function dilateMask(mask, w, h, radius){
    const f = new Float32Array(mask.length);
    for (let i=0;i<mask.length;i++) f[i] = mask[i];
    const b = boxBlur(f, w, h, radius);
    const win = (2*radius+1)*(2*radius+1);
    const out = new Uint8Array(mask.length);
    for (let i=0;i<mask.length;i++) out[i] = b[i] >= 0.5/win ? 1 : 0;
    return out;
  }

  function detectLesionCandidates(canvas, anatomy){
    const { w, h, n, lum, green, red, blue } = imageChannels(canvas);

    const inside = retinaFieldMask(lum, w, h, 0.05);
    let retinaArea = 0;
    for (let i=0;i<n;i++) retinaArea += inside[i];
    if (retinaArea < n*0.05) throw new Error("retinal area too small to analyse");

    const greenS = boxBlur(green, w, h, 1);
    const lumS = boxBlur(lum, w, h, 1);

    const grad = new Float32Array(n);
    for (let y=1;y<h-1;y++){
      for (let x=1;x<w-1;x++){
        const i = y*w+x;
        grad[i] = Math.hypot(lumS[i+1]-lumS[i-1], lumS[i+w]-lumS[i-w]);
      }
    }

    // ---- Lesion evidence: directional top-hats.
    // These decide what counts as a lesion at all. Vessels vanish because they
    // survive a closing along their own direction, the macula vanishes because
    // its gradual ramp barely responds, and a lesion sitting on a vessel is still
    // found because the lesion itself is round regardless of what it touches.
    const seeds = roundStructureSeeds(greenS, lumS, inside, w, h);
    const darkSeed = seeds.darkSeed, brightSeed = seeds.brightSeed;

    // The vessel map comes from the same directional evidence: thin in some
    // direction but not round in all of them. Deriving it here rather than from a
    // separate square closing is both cheaper and sounder, since a square element
    // removes every small lesion along with the vessels.
    const vesselMask = seeds.vesselMask;
    const vesselZone = dilateMask(vesselMask, w, h, 1);

    // ---- Scale saturation.
    // Which rung of the ladder speaks for a region of this size: the first
    // element at least as wide as the region is across. Below that the region has
    // not saturated yet and the ratio would only measure how much of it is still
    // being filled in; at that rung and above, a bounded blob is already complete.
    // Widest structure the ladder can still speak for. Past this the longest
    // element no longer fills the region, so no rung can show saturation and
    // silence must not be read as a verdict — this is what keeps a genuinely
    // large blot hemorrhage from being deleted by a rule that was never about it.
    const satMaxArea = Math.PI*Math.pow(seeds.satSE[seeds.satSE.length-1], 2);

    function satRung(c){
      const r = Math.sqrt(Math.max(c.area,1)/Math.PI);
      for (let k=0; k<seeds.satSE.length-1; k++){
        if (seeds.satSE[k] >= r) return k;
      }
      return -1;                              // wider than the longest element
    }

    // Measured over a disc centred on the region rather than over the region's
    // own pixels, and only where the response clears a core level.
    //
    // Two reasons. The faint margins a region is grown out to carry almost no
    // top-hat response at any element length, so including them drives both means
    // toward zero and the ratio toward noise. More importantly the region's pixel
    // list is not the structure: vessel pixels are removed from what a region may
    // grow into, so a lesion touching a vessel is truncated, and what survives is
    // the part of it furthest from the vessel — its own rim, which is exactly the
    // shape that reads as unbounded. Measured that way the test rejected real
    // lesions as soon as the vessel map became dense enough to bite: on one
    // retina it took five of them. Sampling the image around the region instead
    // keeps the test on the evidence rather than on the bookkeeping.
    function growthRatio(c, dark, k){
      const maps  = dark ? seeds.satDark : seeds.satBright;
      const small = maps[k], big = maps[k+1];
      const core  = (dark ? seeds.tDark : seeds.tBright)*0.5;
      const rad   = Math.max(3, Math.round(Math.sqrt(Math.max(c.area,1)/Math.PI)) + 1);
      const x0 = Math.max(0, Math.round(c.cx)-rad), x1 = Math.min(w-1, Math.round(c.cx)+rad);
      const y0 = Math.max(0, Math.round(c.cy)-rad), y1 = Math.min(h-1, Math.round(c.cy)+rad);
      const r2 = rad*rad;
      let a = 0, b = 0, m = 0;
      for (let y=y0; y<=y1; y++){
        const dy = y - c.cy;
        for (let x=x0; x<=x1; x++){
          const dx = x - c.cx;
          if (dx*dx + dy*dy > r2) continue;
          const i = y*w + x;
          if (!inside[i] || small[i] < core) continue;
          // Vessel pixels are excluded from the measurement, not merely from the
          // region. A lesion sitting beside a vessel is otherwise measured partly
          // on the vessel, and a vessel is the one thing that keeps responding
          // harder as the element grows — so the lesion inherits its neighbour's
          // growth and is rejected for it. Measured on one retina the affected
          // lesions came out between 1.08 and 1.37 against a cut of 1.05, while
          // lesions in open retina sit at 1.00. This is the vessel map being used
          // to protect a finding rather than to suppress one.
          if (vesselMask[i]) continue;
          a += small[i]; b += big[i]; m++;
        }
      }
      if (m < 3 || a <= 0) return null;      // no responding core to judge
      return b/a;
    }
    // A bounded blob is already filled at the element matching its own width, so
    // the next element up adds nothing; anything that continues past the element
    // keeps gaining at every rung. A region wider than the longest element is
    // exempt, because there the test is silent rather than negative — that is
    // what keeps a genuinely large blot hemorrhage from being deleted by a rule
    // that was never about it.
    // Bounded at any scale is bounded. Rather than picking the rung that ought to
    // suit the region's size and trusting that one answer, every rung is asked and
    // the region passes if any of them finds it saturated.
    //
    // Choosing a rung from the region's own area was tried first and is not safe,
    // because the area is not the structure's: vessel pixels are removed from what
    // a region may grow into, so a lesion touching a vessel is truncated and reads
    // as smaller than it is. That sends it to a shorter element than belongs to
    // it, where nothing of its size could have saturated, and it is rejected for
    // failing a test that was asked at the wrong scale. Asking every rung removes
    // the guess: a lesion answers at whichever rung matches it, and a vessel bend
    // answers at none, which is the actual distinction being drawn.
    function failsSaturation(c, dark){
      if (c.area > satMaxArea) return false;
      const k = satRung(c);
      if (k < 0) return false;
      const g = growthRatio(c, dark, k);
      if (g === null) return false;
      return g > SATURATION_MAX_GROWTH;
    }

    // Backgrounds exclude both the black surround and the vessels; a vessel in
    // the averaging window drags the background down and makes ordinary retina
    // between two vessels read as abnormally bright.
    const bgMask = new Uint8Array(n);
    for (let i=0;i<n;i++) bgMask[i] = (inside[i] && !vesselZone[i]) ? 1 : 0;

    // ---- Extent. Seeds say where lesions are; these say how far they reach.
    const scales = [
      Math.max(4,  Math.round(Math.min(w,h)*0.015)),
      Math.max(10, Math.round(Math.min(w,h)*0.055)),
      Math.max(20, Math.round(Math.min(w,h)*0.120))
    ];
    const darkLoose = new Uint8Array(n), brightLoose = new Uint8Array(n);
    const darkDev = new Float32Array(n), brightDev = new Float32Array(n);
    let noiseDark = 1, noiseBright = 1;

    scales.forEach((r, si) => {
      const bgG = maskedBlur(greenS, bgMask, w, h, r);
      const bgL = maskedBlur(lumS, bgMask, w, h, r);
      const dDark = new Float32Array(n), dBright = new Float32Array(n);
      for (let i=0;i<n;i++){
        dDark[i]   = bgG[i] - greenS[i];
        dBright[i] = lumS[i] - bgL[i];
      }
      const sDark = robustSigma(dDark, inside, n);
      const sBright = robustSigma(dBright, inside, n);
      if (si === 0){ noiseDark = sDark; noiseBright = sBright; }
      const tDark   = Math.max(DARK_FLOOR,   sDark*DARK_K)*HYST_RATIO;
      const tBright = Math.max(BRIGHT_FLOOR, sBright*BRIGHT_K)*HYST_RATIO;
      for (let i=0;i<n;i++){
        if (!inside[i]) continue;
        if (dDark[i]   > tDark)   darkLoose[i] = 1;
        if (dBright[i] > tBright) brightLoose[i] = 1;
        if (dDark[i]   > darkDev[i])   darkDev[i] = dDark[i];
        if (dBright[i] > brightDev[i]) brightDev[i] = dBright[i];
      }
    });

    // Growth must not run along a vessel. A lesion lying on one is seeded
    // correctly, but the permissive mask covers the vessel as well, so the region
    // grows out along the whole vessel tree and is then thrown away as too large
    // or too elongated — taking the lesion with it. That is why a lesion touching
    // a vessel could still disappear even once seeding was right. Vessel pixels
    // are therefore removed from what a region is allowed to grow into. The
    // lesion's own pixels are not among them: a lesion answers the directional
    // test strongly, and that is precisely what keeps it out of the vessel map.
    for (let i=0;i<n;i++){
      if (vesselMask[i]){ darkLoose[i] = 0; brightLoose[i] = 0; }
    }
    // A seed must always be able to grow from itself, and a lesion's own body must
    // survive the vessel map crossing it.
    //
    // Restoring only the seeds is not enough once the vessel map is dense and
    // accurate. A lesion lying against a vessel has its seed restored but the rest
    // of its body deleted, so what is left to grow from is the sliver furthest
    // from the vessel — the lesion's own rim, which is the one shape that reads as
    // unbounded, and it is then thrown out by the saturation test. Restoring every
    // pixel that still carries lesion evidence repairs the body instead of just
    // its core. It cannot re-admit the vessel, because this is the roundness
    // response and a vessel barely produces one: that is the whole basis on which
    // the vessel map was built.
    const darkKeep = seeds.tDark*HYST_RATIO, brightKeep = seeds.tBright*HYST_RATIO;
    for (let i=0;i<n;i++){
      if (darkSeed[i] || seeds.roundDark[i] > darkKeep) darkLoose[i] = 1;
      if (brightSeed[i] || seeds.roundBright[i] > brightKeep) brightLoose[i] = 1;
    }

    let gradSum = 0, gradCount = 0;
    for (let i=0;i<n;i++) if (inside[i]){ gradSum += grad[i]; gradCount++; }
    const meanGrad = gradCount ? gradSum/gradCount : 0;
    const sharpEdgeThresh = meanGrad*1.5;

    const maxArea = Math.round(n*0.030);
    const minArea = Math.max(6, Math.round(n*0.000012));
    const maMaxArea = Math.max(minArea+1, Math.round(n*0.00018));
    const cwsMinArea = Math.round(n*CWS_MIN_AREA_FRAC);

    const discR = anatomy ? anatomy.disc.radius : 0;
    function nearDisc(x, y, mult){
      if (!anatomy) return false;
      return Math.hypot(x-anatomy.disc.x, y-anatomy.disc.y) <= discR*mult;
    }

    // ---- Peripapillary bright suppression, part one: measured disc tissue.
    //
    // The dimensions are checked rather than assumed. Landmarks and detection run
    // on the same canvas today, but a mask indexed against a different size would
    // not fail loudly — it would quietly delete findings somewhere else in the
    // image, which is the worst way for this to go wrong.
    const discTissue = (anatomy && anatomy.disc.tissue &&
                        anatomy.disc.tissueW === w && anatomy.disc.tissueH === h)
                       ? anatomy.disc.tissue : null;

    function discTissueFrac(c){
      if (!discTissue) return 0;
      let on = 0;
      for (let i=0;i<c.pixels.length;i++) if (discTissue[c.pixels[i]]) on++;
      return c.pixels.length ? on/c.pixels.length : 0;
    }

    // ---- Part two: colour.
    //
    // Past the halo there is still nerve-fibre reflex and atrophic mottling, and
    // in shape, size and contrast those are indistinguishable from a hard
    // exudate — which is why every geometric rule tried here left some of them
    // standing. What separates them is what they are made of. An exudate is lipid
    // and reads yellow: it gains in red and green and hardly at all in blue.
    // Disc tissue, sclera and reflex are white or grey and gain in all three, so
    // their blue fraction climbs while an exudate's stays near the retina's.
    //
    // Both ends of the scale are read off this image — disc tissue for white,
    // surrounding retina for the other end — so nothing here depends on the
    // camera's white balance, and an image where the two ends do not separate is
    // one where the test says nothing rather than one where it guesses.
    function blueFraction(i){
      const sum = red[i] + green[i] + blue[i];
      return sum > 24 ? blue[i]/sum : -1;    // too dark to carry a readable hue
    }

    const discColour = (function(){
      if (!anatomy || discR <= 0) return null;
      const onDisc = [], onRetina = [];
      const reach = discR*PERIPAPILLARY_MULT*1.6;
      const cx = anatomy.disc.x, cy = anatomy.disc.y;
      const reach2 = reach*reach, far2 = (discR*1.6)*(discR*1.6);
      const core2 = discR*discR*0.64;
      for (let y=Math.max(0,Math.round(cy-reach)); y<=Math.min(h-1,Math.round(cy+reach)); y++){
        const dy = y-cy;
        for (let x=Math.max(0,Math.round(cx-reach)); x<=Math.min(w-1,Math.round(cx+reach)); x++){
          const dx = x-cx, d2 = dx*dx + dy*dy;
          if (d2 > reach2) continue;
          const i = y*w+x;
          if (!inside[i]) continue;
          const f = blueFraction(i);
          if (f < 0) continue;
          // White reference: measured disc tissue where the extent could be
          // grown, the inner core of the circle otherwise — that core is disc
          // whether or not the radius around it was right.
          if (discTissue ? discTissue[i] : d2 <= core2){ onDisc.push(f); continue; }
          // Retina reference: outside the rim and off the vessels, so neither the
          // disc's own edge nor a vessel's blood colour sets the other end.
          if (d2 > far2 && !(discTissue && discTissue[i]) && !vesselZone[i]) onRetina.push(f);
        }
      }
      if (onDisc.length < 200 || onRetina.length < 200) return null;
      const discF = medianOf(onDisc), retinaF = medianOf(onRetina);
      if (discF - retinaF < DISC_COLOUR_MIN_SEPARATION) return null;
      return { discF, retinaF, cut: retinaF + (discF-retinaF)*DISC_COLOUR_LEVEL };
    })();

    function looksLikeDiscTissue(c){
      if (!discColour) return false;
      let sum = 0, m = 0;
      for (let i=0;i<c.pixels.length;i++){
        const f = blueFraction(c.pixels[i]);
        if (f < 0) continue;
        sum += f; m++;
      }
      if (m < 3) return false;
      return (sum/m) >= discColour.cut;
    }
    const hasMacula = !!(anatomy && anatomy.fovea);
    const ap = apertureGeometry(lum, w, h);
    const edgeR2 = (ap.R*EDGE_ZONE_FRAC)*(ap.R*EDGE_ZONE_FRAC);
    const edgeArtifactArea = Math.round(n*EDGE_ARTIFACT_AREA_FRAC);
    function isEdgeArtifact(c){
      if (c.area < edgeArtifactArea) return false;      // small peripheral lesions stay
      const dx = c.cx-ap.cx, dy = c.cy-ap.cy;
      return (dx*dx + dy*dy) > edgeR2;
    }

    // The junction test is only meaningful at the scale vessels cross and branch
    // at. A large blot hemorrhage genuinely does have several vessels running out
    // of the area around it, and testing one would throw away a real and serious
    // finding, so anything wider than a couple of vessel widths is exempt.
    const seLen = Math.max(3, Math.round(Math.min(w,h)*LINEAR_SE_FRAC));
    const junctionMaxArea = Math.round(Math.PI*(seLen*2.0)*(seLen*2.0));
    // How far to walk when looking for vessel arms: about one and a half
    // vessel elements, which outlasts any lesion collar.
    const junctionArmLen = seLen*1.5;

    const candidates = [];
    const rejected = { vessel:0, junction:0, tooLarge:0, tooSmall:0, disc:0, discTissue:0, discColour:0, weak:0, streak:0, noSeed:0, macula:0, smooth:0, edge:0, unbounded:0 };

    const darkCovered = new Uint8Array(n), brightCovered = new Uint8Array(n);

    function vesselOverlapFrac(c){
      let on = 0;
      for (let i=0;i<c.pixels.length;i++) if (vesselZone[c.pixels[i]]) on++;
      return c.pixels.length ? on/c.pixels.length : 0;
    }

    function classify(comps, dark){
      const covered = dark ? darkCovered : brightCovered;
      comps.forEach(c => {
        // No directional response anywhere in the region means no lesion: this
        // is what excludes the vessels, the macula and smooth background shading,
        // all in one test, without needing to know where they are.
        if (!c.hasSeed){ rejected.noSeed++; return; }
        if (c.area < minArea){ rejected.tooSmall++; return; }
        if (c.area > maxArea){ rejected.tooLarge++; return; }
        if (nearDisc(c.cx, c.cy, 1.15)){ rejected.disc++; return; }
        // Outside that circle the disc can still be the explanation for a bright
        // mark: the rim it was grown from, a crescent of sclera beside it, the
        // nerve-fibre reflex arcing off it. Two independent ways of saying so,
        // either sufficient — the mark is built out of measured disc tissue, or
        // it is the colour of disc tissue rather than the colour of exudate.
        //
        // Dark candidates are deliberately exempt. Nothing about the disc is
        // dark, so a dark mark beside it is a hemorrhage and has to survive; and
        // the disc's own margin is where a disc hemorrhage sits, which is a
        // finding worth more than everything this rule removes.
        if (!dark && nearDisc(c.cx, c.cy, PERIPAPILLARY_MULT)){
          if (discTissueFrac(c) >= DISC_TISSUE_OVERLAP){ rejected.discTissue++; return; }
          if (looksLikeDiscTissue(c)){ rejected.discColour++; return; }
        }
        if (isEdgeArtifact(c)){ rejected.edge++; return; }
        if (c.aspect > 4.0 && c.fillRatio < 0.30){ rejected.streak++; return; }

        // A region lying almost entirely on the vessel map is a piece of vessel,
        // whatever seeded it. The threshold is high on purpose: a lesion touching
        // a vessel overlaps it partially, and must survive.
        if (vesselOverlapFrac(c) > VESSEL_OVERLAP_REJECT){ rejected.vessel++; return; }

        // Crossings, bifurcations, bends and choroidal mottling answer the
        // roundness test at one element length the same way a lesion does. They
        // are told apart by whether the response stops growing when the element
        // does: a lesion is bounded, they are not.
        if (failsSaturation(c, dark)){ rejected.unbounded++; return; }

        // Crossings, bifurcations and tight bends answer the roundness test the
        // same way a lesion does. They are told apart by what runs out of them.
        let onVessel = false;
        if (dark && c.area <= junctionMaxArea){
          const ctx = vesselArms(c.cx, c.cy, c.area, vesselZone, w, h, junctionArmLen);
          if (isVesselJunction(ctx)){ rejected.junction++; return; }
          onVessel = isThroughVessel(ctx) || ctx.arms >= 1;
        }

        const floor = dark ? DARK_CONTRAST_FLOOR : BRIGHT_CONTRAST_FLOOR;
        const noise = dark ? noiseDark : noiseBright;
        // Contrast is measured against a background that excludes vessels, so a
        // vessel scores well on it simply for being a vessel. A candidate on the
        // vasculature therefore has to clear a higher bar than one in open
        // retina before the same figure means the same thing.
        const bar = Math.max(floor, noise*MIN_CONTRAST_K) * (onVessel ? ON_VESSEL_CONTRAST_FACTOR : 1);
        if (c.maxContrast < bar){ rejected.weak++; return; }

        let type;
        if (dark){
          type = (c.area <= maMaxArea && c.aspect <= 2.1 && c.fillRatio >= 0.45)
            ? LESION_TYPES.MA : LESION_TYPES.HEM;
        } else if (c.meanGradient >= sharpEdgeThresh){
          type = LESION_TYPES.HE;
        } else if (c.area >= cwsMinArea){
          type = LESION_TYPES.CWS;
        } else {
          rejected.weak++; return;
        }
        candidates.push(Object.assign(c, { type, onVessel }));
        for (let i=0;i<c.pixels.length;i++) covered[c.pixels[i]] = 1;
      });
    }

    classify(connectedComponents(darkLoose, w, h, maxArea, darkDev, grad, darkSeed), true);
    classify(connectedComponents(brightLoose, w, h, maxArea, brightDev, grad, brightSeed), false);

    // Safety net. Something that was seeded must not vanish because the region it
    // grew into failed a size or shape test; the evidence for the lesion was the
    // seed, not the extent. Any seed not covered by an accepted region is
    // reconsidered on its own extent, so the worst case is that a lesion is
    // reported slightly smaller than it really is, never that it is reported at
    // all. Seeds cannot reintroduce vessels, because vessels never seed.
    const darkLeft = new Uint8Array(n), brightLeft = new Uint8Array(n);
    let dLeft = 0, bLeft = 0;
    for (let i=0;i<n;i++){
      if (darkSeed[i] && !darkCovered[i]){ darkLeft[i] = 1; dLeft++; }
      if (brightSeed[i] && !brightCovered[i]){ brightLeft[i] = 1; bLeft++; }
    }
    if (dLeft) classify(connectedComponents(darkLeft, w, h, maxArea, darkDev, grad, null), true);
    if (bLeft) classify(connectedComponents(brightLeft, w, h, maxArea, brightDev, grad, null), false);

    candidates.sort((a,b) => b.area - a.area);

    const counts = {};
    LESION_ORDER.forEach(k => counts[k] = 0);
    let lesionPixels = 0, onVesselCount = 0;
    candidates.forEach(c => {
      counts[c.type.key]++; lesionPixels += c.area;
      if (c.onVessel) onVesselCount++;
    });

    return {
      candidates, counts, rejected, w, h,
      retinaArea, lesionPixels, onVesselCount,
      noise: { dark: noiseDark, bright: noiseBright },
      scales,
      discExcluded: !!anatomy,
      discExtentMeasured: !!(anatomy && anatomy.disc.measuredExtent),
      discTissueMask: !!discTissue,
      discColourUsed: !!discColour,
      maculaExcluded: hasMacula
    };
  }


  // ---------------------------------------------------------------------
  // ICDR SEVERITY ESTIMATE
  //
  // The published scale is defined by which lesion types are present and, for
  // severe disease, by how many hemorrhages appear in each quadrant:
  //
  //   0  No apparent retinopathy   no abnormality
  //   1  Mild NPDR                 microaneurysms only
  //   2  Moderate NPDR             more than microaneurysms, less than severe
  //   3  Severe NPDR               no PDR signs, and any of the 4-2-1 rule:
  //                                  >20 intraretinal hemorrhages in each of 4 quadrants,
  //                                  venous beading in 2 or more quadrants,
  //                                  prominent IRMA in 1 or more quadrant
  //   4  Proliferative DR          neovascularisation, or vitreous/preretinal hemorrhage
  //
  // Applying it here is legitimate because the rule is itself a function of lesion
  // type and count, which is exactly what Step 4 produces. What it is NOT is a
  // diagnosis: the counts are unvalidated candidates, so the grade inherits every
  // one of their errors.
  //
  // There is also a hard ceiling. Two of the three arms of the 4-2-1 rule are
  // venous beading and IRMA, and level 4 is defined by neovascularisation. This
  // detector cannot see any of the three, because all are elongated structures that
  // its vessel test removes by construction. So it can reach level 3 only through
  // the hemorrhage arm, can never reach level 4, and above all can never rule
  // either of them out. That limitation is stated with every result rather than
  // buried, because a low grade here does not mean a low grade in the eye.
  // ---------------------------------------------------------------------
  function quadrantLesionCounts(lesions, anatomy){
    if (!anatomy || !lesions) return null;
    const grid = {};
    LESION_ORDER.forEach(k => { grid[k] = {}; QUADRANT_NAMES.forEach(q => grid[k][q] = 0); });
    lesions.candidates.forEach(c => {
      const q = quadrantLabel(c.cx, c.cy, anatomy.disc, anatomy.nasalSide);
      if (grid[c.type.key] && grid[c.type.key][q] !== undefined) grid[c.type.key][q]++;
    });
    return grid;
  }

  function assessSeverity(lesions, anatomy, grading, quality){
    const counts = lesions ? lesions.counts : null;
    const ma  = counts ? counts.MA  : 0;
    const hem = counts ? counts.HEM : 0;
    const he  = counts ? counts.HE  : 0;
    const cws = counts ? counts.CWS : 0;
    const beyondMA = hem + he + cws;
    const total = ma + beyondMA;

    const grid = quadrantLesionCounts(lesions, anatomy);
    let quadsOverThreshold = 0;
    if (grid) QUADRANT_NAMES.forEach(q => {
      if (grid.HEM[q] > ICDR_HEM_PER_QUADRANT) quadsOverThreshold++;
    });
    const hemArmMet = !!grid && quadsOverThreshold === 4;

    let level, label, basis = [];
    if (!lesions){
      level = null; label = "Not assessed";
      basis.push("lesion detection did not run on this image");
    } else if (hemArmMet){
      level = 3; label = "Severe NPDR pattern";
      basis.push("more than " + ICDR_HEM_PER_QUADRANT + " hemorrhage candidates in each of the four quadrants, which is the hemorrhage arm of the 4-2-1 rule");
    } else if (beyondMA > 0){
      level = 2; label = "Moderate NPDR pattern";
      basis.push("more than microaneurysms alone: " + hem + " hemorrhage, " + he + " hard exudate and " + cws + " cotton wool candidates");
    } else if (ma > 0){
      level = 1; label = "Mild NPDR pattern";
      basis.push(ma + " microaneurysm candidate" + (ma===1?"":"s") + " and nothing else");
    } else {
      level = 0; label = "No retinopathy observed";
      basis.push("no lesion candidate passed the detector on this image");
    }

    // Everything that makes this estimate untrustworthy on this particular image.
    const doubts = [];
    if (!lesions) doubts.push("lesion detection did not complete");
    if (!anatomy) doubts.push("landmarks could not be estimated, so the quadrant rule for severe disease could not be applied at all");
    if (quality && quality.verdict && quality.verdict.verdict === "enhance")
      doubts.push("image quality was borderline and had to be enhanced before analysis");
    if (lesions && total > NOISE_SUSPICION_COUNT)
      doubts.push("the candidate count is high enough to suggest the detector is responding to image noise");
    if (grid && quadsOverThreshold > 0 && quadsOverThreshold < 4)
      doubts.push("hemorrhage candidates exceed the severe-disease threshold in " + quadsOverThreshold + " of four quadrants, which sits right on the boundary of the rule");

    // The cross-check that the two independent stages exist to provide.
    if (grading){
      const modelSaysDR = grading.predClass === 1;
      if (modelSaysDR && level === 0)
        doubts.push("the classifier reports disease present while the detector found no lesion at all, and the two disagree");
      if (!modelSaysDR && level >= 2)
        doubts.push("the classifier reports no disease while the detector found lesions beyond microaneurysms, and the two disagree");
    } else {
      doubts.push("the classifier did not run, so there is no independent check on this result");
    }

    // Refer whenever the rule says referable, whenever anything is in doubt, and
    // whenever any lesion at all was seen. Only a clean, agreeing, lesion-free
    // image avoids it.
    const referable = level !== null && level >= 2;
    const uncertain = doubts.length > 0;
    const refer = referable || uncertain || (level !== null && level >= 1);

    let reason;
    if (referable) reason = "This image reaches the referable threshold (moderate NPDR or worse) under the ICDR rule.";
    else if (uncertain) reason = "This result is not reliable enough to stand on its own.";
    else if (level >= 1) reason = "Lesions were seen. Any retinopathy needs a specialist opinion.";
    else reason = "Severe disease and proliferative disease cannot be excluded by this method.";

    return {
      level, label, basis, doubts, refer, referable, uncertain, reason,
      grid, quadsOverThreshold, counts, total
    };
  }

  function renderSeverity(sev){
    const box = document.getElementById("grade-result");
    const scale = [
      [0,"0","No DR"],[1,"1","Mild"],[2,"2","Moderate"],[3,"3","Severe"],[4,"4","Proliferative"]
    ].map(([n,num,txt]) =>
      '<div class="' + (sev.level === n ? "on" : "") + '">' + num + '<br>' + txt + '</div>'
    ).join("");

    let html = '<div class="grade-box">';
    html += '<p class="grade-level">' + (sev.level === null ? "Not assessed" : "Level " + sev.level) + '</p>';
    html += '<p class="grade-label">' + sev.label + (sev.referable ? " &middot; referable" : "") + '</p>';
    html += '<div class="grade-scale">' + scale + '</div>';
    html += '<ul class="grade-basis">' + sev.basis.map(b => "<li>" + b + "</li>").join("") + '</ul>';

    html += '<div class="ceiling"><strong>What this grade cannot say.</strong> Two of the three arms of the ' +
      'severe-disease rule are venous beading and IRMA, and proliferative disease is defined by ' +
      'neovascularisation. This detector cannot see any of the three, because all are elongated and its ' +
      'vessel test removes them by construction. It can therefore reach level 3 only through the hemorrhage ' +
      'count, can never reach level 4, and <strong>can never rule either of them out</strong>. A low grade ' +
      'here does not mean a low grade in the eye.</div>';

    if (sev.doubts.length){
      html += '<div class="ceiling"><strong>Why this particular result is doubtful.</strong><ul>' +
        sev.doubts.map(d => "<li>" + d + "</li>").join("") + '</ul></div>';
    }
    html += '</div>';
    box.innerHTML = html;

    // quadrant table for the hemorrhage arm
    const qwrap = document.getElementById("grade-quadrants");
    if (!sev.grid){
      qwrap.innerHTML = '<p class="small">Quadrant counts are unavailable because landmark estimation did not produce an optic disc position, so the 4-2-1 rule could not be evaluated.</p>';
    } else {
      const rows = QUADRANT_NAMES.map(q =>
        "<tr><td>" + q + "</td><td>" + sev.grid.HEM[q] + "</td><td>" +
        (sev.grid.HEM[q] > ICDR_HEM_PER_QUADRANT ? "over threshold" : "under") + "</td></tr>"
      ).join("");
      qwrap.innerHTML =
        '<h3>Hemorrhage candidates per quadrant</h3>' +
        '<table class="icdr-table"><thead><tr><th>Quadrant</th><th>Candidates</th><th>Against the ' +
        ICDR_HEM_PER_QUADRANT + '-per-quadrant threshold</th></tr></thead><tbody>' + rows + '</tbody></table>' +
        '<p class="small">Severe NPDR needs all four quadrants over the threshold; ' + sev.quadsOverThreshold +
        ' of four are here. Quadrants are measured from the estimated optic disc rather than a clinical ' +
        'landmark set, so a mislocated disc moves every count between quadrants.</p>';
    }
  }

  // Raises the referral banner without a grade behind it, for the cases where the
  // app could not assess the image at all. Those are the moments the user is told
  // least, so they are exactly the moments the warning matters most.
  function referWithoutGrade(reason, points){
    const el = document.getElementById("refer-banner");
    const txt = document.getElementById("refer-reason");
    if (!el || !txt) return;
    let html = reason;
    if (points && points.length) html += "<ul>" + points.map(p => "<li>" + p + "</li>").join("") + "</ul>";
    txt.innerHTML = html;
    el.classList.add("visible");
  }

  function renderReferBanner(sev){
    const el = document.getElementById("refer-banner");
    const txt = document.getElementById("refer-reason");
    if (!el || !txt) return;
    if (!sev || !sev.refer){ el.classList.remove("visible"); return; }
    let html = sev.reason;
    if (sev.doubts.length){
      html += "<ul>" + sev.doubts.slice(0,3).map(d => "<li>" + d + "</li>").join("") + "</ul>";
    }
    txt.innerHTML = html;
    el.classList.add("visible");
  }

  // ---------------------------------------------------------------------
  // STEP 4 RENDERING
  // ---------------------------------------------------------------------
  function renderBinaryMask(lesions, targetCanvas){
    const { w, h, candidates } = lesions;
    targetCanvas.width = w; targetCanvas.height = h;
    const ctx = targetCanvas.getContext("2d");
    const img = ctx.createImageData(w, h);
    const px = img.data;
    for (let i=0, p=0; p<w*h; i+=4, p++){
      px[i] = 0; px[i+1] = 0; px[i+2] = 0; px[i+3] = 255;
    }
    candidates.forEach(c => {
      c.pixels.forEach(idx => {
        const i = idx*4;
        px[i] = 255; px[i+1] = 255; px[i+2] = 255;
      });
    });
    ctx.putImageData(img, 0, 0);
  }

  function renderTypedMask(lesions, targetCanvas){
    const { w, h, candidates } = lesions;
    targetCanvas.width = w; targetCanvas.height = h;
    const ctx = targetCanvas.getContext("2d");
    const img = ctx.createImageData(w, h);
    const px = img.data;
    for (let i=0, p=0; p<w*h; i+=4, p++){
      px[i] = 0; px[i+1] = 0; px[i+2] = 0; px[i+3] = 255;
    }
    candidates.forEach(c => {
      const g = c.type.grey;
      c.pixels.forEach(idx => {
        const i = idx*4;
        px[i] = g; px[i+1] = g; px[i+2] = g;
      });
    });
    ctx.putImageData(img, 0, 0);
  }

  // Candidates are usually a few pixels across, too small to see against the
  // fundus, so each is ringed at a legible minimum radius and the largest few
  // are labelled with their type.
  function renderLesionOverlay(sourceCanvas, lesions, targetCanvas){
    const w = sourceCanvas.width, h = sourceCanvas.height;
    targetCanvas.width = w; targetCanvas.height = h;
    const ctx = targetCanvas.getContext("2d");
    ctx.drawImage(sourceCanvas, 0, 0);

    const fontPx = Math.max(9, Math.round(w*0.022));
    const labelLimit = 3;
    // Candidates are sorted largest-first. Ringing thousands of them would be
    // both slow and unreadable, so the overlay caps what it draws; the two mask
    // canvases still show every candidate pixel.
    const drawn = lesions.candidates.slice(0, OVERLAY_CIRCLE_LIMIT);
    // Label the largest few of each type rather than the largest overall, so no
    // type ends up entirely unlabelled just because its regions are small.
    const labelledPerType = {};
    drawn.forEach(c => {
      const r = Math.max(5, Math.sqrt(c.area/Math.PI)*1.9);
      drawLesionMarker(ctx, c.type.shape, c.cx, c.cy, r);
      // A candidate lying along a vessel gets a tick through its marker. It is
      // still a candidate — a microaneurysm beside a venule is real — but a
      // reader deciding whether a mark is a lesion or a wide spot in the vessel
      // it sits on needs to be told which it is, and the mask alone cannot say.
      if (c.onVessel){
        ctx.save();
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(c.cx-r*0.75, c.cy+r*0.75); ctx.lineTo(c.cx+r*0.75, c.cy-r*0.75); ctx.stroke();
        ctx.strokeStyle = "#000"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(c.cx-r*0.75, c.cy+r*0.75); ctx.lineTo(c.cx+r*0.75, c.cy-r*0.75); ctx.stroke();
        ctx.restore();
      }
      const seen = labelledPerType[c.type.key] || 0;
      if (seen < labelLimit){
        labelledPerType[c.type.key] = seen+1;
        haloText(ctx, c.type.key, c.cx, clamp(c.cy - r - 2, fontPx, h-2), "center", "bottom", true, fontPx);
      }
    });
    return { drawn: drawn.length, total: lesions.candidates.length };
  }

  function renderLesionTables(lesions, anatomy){
    const total = lesions.candidates.length;

    const legend = document.getElementById("lesion-legend");
    legend.innerHTML = LESION_ORDER.map(k => {
      const t = LESION_TYPES[k];
      const g = `rgb(${t.grey},${t.grey},${t.grey})`;
      return `<div class="legend-row"><span class="swatch" style="background:${g}"></span>${t.label} — <strong>${lesions.counts[k]}</strong> · drawn as a ${t.shapeName} on the fundus overlay, labelled <code>${t.key}</code></div>`;
    }).join("") +
      `<div class="legend-row"><span class="swatch" style="background:transparent;position:relative">` +
      `<span style="position:absolute;left:-1px;top:6px;width:15px;height:1px;background:#000;` +
      `display:block;transform:rotate(-45deg)"></span></span>` +
      `A marker struck through — <strong>${lesions.onVesselCount || 0}</strong> · the candidate lies along the course of a vessel. ` +
      `It is kept, because a microaneurysm beside a venule is a real and common finding, but it is held to a stricter contrast bar and flagged here because a wide spot in a vessel looks the same.</div>`;

    const summary = document.getElementById("lesion-summary");
    const pctOfRetina = lesions.retinaArea ? (lesions.lesionPixels/lesions.retinaArea)*100 : 0;
    if (total === 0){
      summary.innerHTML = `<p class="notice">No candidate regions passed the filters on this image. That is not the same as a normal retina — small or low-contrast lesions can fall below the adaptive threshold, and the classifier in Step 2 reached its own conclusion independently of this result.</p>`;
    } else {
      const noisy = total > NOISE_SUSPICION_COUNT;
      const capped = total > OVERLAY_CIRCLE_LIMIT;
      summary.innerHTML = `<p><strong>${total} candidate region${total===1?"":"s"}</strong> covering ${lesions.lesionPixels.toLocaleString()} px, ${pctOfRetina.toFixed(2)}% of the retinal area.
        Rejected during filtering: ${lesions.rejected.vessel} as vessel or vessel fragment, ${lesions.rejected.junction} as vessel crossings, bifurcations or bends, ${lesions.rejected.unbounded} as unbounded structures whose response kept growing with the measuring element, ${lesions.rejected.weak} as too faint against the noise floor, ${lesions.rejected.tooLarge} as too large, ${lesions.rejected.tooSmall} as too small, ${lesions.rejected.streak} as bright streaks, ${lesions.rejected.disc} inside the optic disc, ${lesions.rejected.discTissue} as bright regions built out of measured disc tissue, ${lesions.rejected.discColour} as peripapillary marks the colour of disc or sclera rather than of exudate, ${lesions.rejected.edge} as large regions at the edge of the aperture, ${lesions.rejected.macula} as macular pigmentation, ${lesions.rejected.smooth} as broad smooth shading, ${lesions.rejected.noSeed} for having no core strong enough to seed a region.
        ${lesions.onVesselCount ? `<strong>${lesions.onVesselCount}</strong> of them lie along a vessel and are struck through on the overlay — treat those with extra caution.` : ""}
        ${capped ? `The fundus overlay rings the ${OVERLAY_CIRCLE_LIMIT} largest; both mask canvases show all of them.` : ""}
        ${lesions.discExcluded ? "" : "<em>Optic disc position was unavailable, so the disc was not excluded and its bright pixels may appear as exudate candidates.</em>"}
        ${lesions.discExcluded && !lesions.discExtentMeasured ? "<em>The disc's extent could not be grown from its own brightness on this image, so only the estimated circle excluded it — bright marks at its margin may have survived.</em>" : ""}
        ${lesions.discExcluded && lesions.discExtentMeasured && !lesions.discColourUsed ? "<em>Disc and retina did not separate in colour on this image, so peripapillary marks were judged on the measured disc extent alone.</em>" : ""}
        ${lesions.maculaExcluded ? "" : "<em>Macula position was unavailable, so normal macular darkening may appear here as a large dark candidate.</em>"}</p>`
        + (noisy ? `<p class="notice"><strong>This count is too high to be lesions.</strong> ${total} candidates on one image almost always means the detector is firing on image noise, compression artefacts, or texture — the adaptive threshold drops with the image's own contrast, so a grainy photo produces hundreds of spurious specks. Do not read these counts as a lesion burden on this image.</p>` : "");
    }

    const quadWrap = document.getElementById("lesion-quadrants");
    if (!anatomy || total === 0){
      quadWrap.innerHTML = anatomy ? "" :
        `<p class="small">Quadrant counts are unavailable because landmark estimation did not produce an optic disc position.</p>`;
      return;
    }

    const quadNames = ["Superior-Nasal","Superior-Temporal","Inferior-Nasal","Inferior-Temporal"];
    const grid = {};
    LESION_ORDER.forEach(k => { grid[k] = {}; quadNames.forEach(q => grid[k][q] = 0); });
    lesions.candidates.forEach(c => {
      const q = quadrantLabel(c.cx, c.cy, anatomy.disc, anatomy.nasalSide);
      if (grid[c.type.key] && grid[c.type.key][q] !== undefined) grid[c.type.key][q]++;
    });

    let rows = LESION_ORDER.map(k => {
      const cells = quadNames.map(q => `<td>${grid[k][q]}</td>`).join("");
      return `<tr><td>${LESION_TYPES[k].label}</td>${cells}<td><strong>${lesions.counts[k]}</strong></td></tr>`;
    }).join("");

    quadWrap.innerHTML = `
      <h3>Candidates per retinal quadrant</h3>
      <table class="icdr-table">
        <thead><tr><th>Type</th>${quadNames.map(q=>`<th>${q}</th>`).join("")}<th>Total</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="small">The ICDR severity-3 rule counts hemorrhages by quadrant, so this table is laid out the same way. It is <strong>not</strong> a 4:2:1 assessment: that rule counts confirmed intraretinal hemorrhages, these are unvalidated candidates, and the quadrant axes come from an estimated disc position. Reading a severity grade off this table would compound three separate approximations.</p>`;
  }

  // ---------------------------------------------------------------------
  // SMALL RENDER HELPERS
  // ---------------------------------------------------------------------
  function renderBarRow(container, label, value, maxValue, unit){
    const row = document.createElement("div");
    row.className = "bar-row";
    const pct = maxValue > 0 ? clamp((value/maxValue)*100, 0, 100) : 0;
    row.innerHTML = `
      <div class="bar-label">${label}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      <div class="bar-value">${fmt(value,0)}${unit||""}</div>
    `;
    container.appendChild(row);
  }

  // ---------------------------------------------------------------------
  // RENDER: STEP 1
  // ---------------------------------------------------------------------
  function renderQualityStep(metrics, verdict){
    const T = QUALITY_THRESHOLDS;
    const table = $("#quality-table");
    table.innerHTML = `
      <tr><td>Focus (Laplacian variance)</td><td>${metrics.focusVar.toFixed(1)} <span class="small">(reject below ${T.minFocusVar}, enhance below ${T.enhanceFocusVar})</span></td></tr>
      <tr><td>Brightness mean (0–255)</td><td>${metrics.brightMean.toFixed(1)} <span class="small">(acceptable range ${T.minBrightMean}–${T.maxBrightMean})</span></td></tr>
      <tr><td>Brightness spread (std dev)</td><td>${metrics.brightStd.toFixed(1)}</td></tr>
      <tr><td>Field-of-view coverage</td><td>${(metrics.fov*100).toFixed(0)}% of frame <span class="small">(reject below ${(T.minFov*100).toFixed(0)}%)</span></td></tr>
    `;
    const notice = $("#quality-notice");
    if (verdict.verdict === "reject"){
      notice.innerHTML = `<div class="notice reject"><strong>Rejected — recapture required.</strong> ${verdict.reason}</div>`;
    } else if (verdict.verdict === "enhance"){
      notice.innerHTML = `<div class="notice">Borderline quality (${verdict.reasons.join(", ")}) — real enhancement applied: light denoising, illumination normalization, and CLAHE-style local contrast enhancement (8×8 tiles, clipped/redistributed histogram, bilinearly-interpolated tile boundaries). Proceeding to classification on the enhanced image.</div>`;
    } else {
      notice.innerHTML = `<div class="notice">Image quality acceptable — no enhancement needed.</div>`;
    }
  }

  // ---------------------------------------------------------------------
  // RENDER: STEP 2
  // ---------------------------------------------------------------------
  function renderGradingStep(result){
    const scopeNotice = $("#model-scope-notice");
    scopeNotice.innerHTML = `<strong>Model scope:</strong> the loaded classifier is a real MobileNet binary detector (Normal vs. DR present) trained on the APTOS 2019 dataset — it does not output a validated 5-class ICDR 0–4 severity grade. The probabilities below are genuine softmax output from that model, not simulated. Any image flagged "DR present" should be treated as referable and forwarded for full ICDR grading by a specialist; this app does not claim to distinguish mild from moderate/severe/proliferative disease.`;

    const results = $("#grading-results");
    const bars = document.createElement("div");
    CLASS_NAMES.forEach((name, i) => {
      renderBarRow(bars, name, result.probs[i]*100, 100, "%");
    });
    const referable = result.predClass === 1;
    const badge = document.createElement("p");
    badge.innerHTML = `<span class="referable-badge ${referable?"":"negative"}">${referable ? "Referable — DR detected" : "Non-referable — no DR detected"}</span>
      <br><span class="small">Predicted class: ${CLASS_NAMES[result.predClass]} (argmax of the two real softmax outputs, threshold 0.5). Inference time: ${result.elapsedMs.toFixed(0)} ms.</span>`;
    results.innerHTML = "";
    results.appendChild(bars);
    results.appendChild(badge);

    // Softmax saturation is normal for a binary CNN like this one and is routinely
    // misread as certainty, so the caveat is stated wherever the number appears.
    const topProb = Math.max(result.probs[0], result.probs[1]);
    const calib = document.createElement("p");
    calib.className = "notice";
    calib.innerHTML = topProb > 0.99
      ? `<strong>On that ${(topProb*100).toFixed(1)}% reading.</strong> This is a saturated softmax output, not a calibrated probability. It means the image sits far on one side of the model's decision boundary — nothing more. It is not a ${(topProb*100).toFixed(0)}% chance that the diagnosis is correct, and a confidently wrong answer looks exactly like this one. The model has not been calibrated against a held-out set in this app, and it reports the same way on images unlike anything in its training data. Read it as a direction, not a degree of certainty.`
      : `<strong>On confidence figures.</strong> These are raw softmax outputs, not calibrated probabilities. They indicate which side of the model's decision boundary the image falls on and how far, not the likelihood that a diagnosis is correct. No calibration was performed in this app.`;
    results.appendChild(calib);
  }

  // ---------------------------------------------------------------------
  // RENDER: STEP 3
  // ---------------------------------------------------------------------
  // Draws both Grad-CAM panels and their region markers. Shared so the controls
  // can redraw without recomputing the gradient.
  function redrawCamRegions(sourceCanvas, cam, anatomy, desc){
    const gcCanvas = document.getElementById("canvas-gradcam");
    const gcColour = document.getElementById("canvas-gradcam-colour");
    renderGradCAMOverlay(sourceCanvas, cam.camArray, cam.h, cam.w, gcCanvas, cam.inside, cam.feather);
    renderGradCAMColour(sourceCanvas, cam.camArray, gcColour, cam.inside, cam.feather);
    drawFieldBoundary(gcCanvas, cam.field);
    drawFieldBoundary(gcColour, cam.field);
    const blobs = findDefectBlobs(cam.camArray, cam.h, cam.w, BLOB_THRESHOLD, BLOB_MIN_AREA_FRAC, cam.inside, cam.stats);
    const marked = drawDefectMarkers(gcCanvas, blobs, anatomy);
    drawDefectMarkers(gcColour, blobs, anatomy);
    state.lastFindings = marked.findings;
    renderFindingsLegend(marked.findings, marked.discCount);
    const t = document.getElementById("blob-threshold-text");
    if (t) t.textContent = Math.round(BLOB_THRESHOLD*100) + "%";
    const ln = document.getElementById("cam-layer-name");
    if (ln && state.camLayerName) ln.textContent = state.camLayerName;
    return marked;
  }

  // The edge of the region the map was measured and scaled over. Drawn dashed
  // and labelled, in the same idiom the app uses everywhere else for something
  // estimated rather than known.
  function drawFieldBoundary(targetCanvas, field){
    if (!field) return;
    const ctx = targetCanvas.getContext("2d");
    const fontPx = Math.max(10, Math.round(targetCanvas.width*0.022));
    strokeCircle(ctx, field.cx, field.cy, field.r, [6,5]);
    haloText(ctx, "analysed field", field.cx,
      clamp(field.cy - field.r - 3, fontPx, targetCanvas.height-2),
      "center", "bottom", false, fontPx);
  }

  function renderGradCAMStep(desc, findings, anatomy){
    const note = $("#gradcam-note");
    const focalText = desc.focal
      ? `focal, concentrated in a small area (top 10% of pixels account for ${(desc.concentration*100).toFixed(0)}% of total activation). A focal hotspot in this pattern can be consistent with a localized lesion cluster such as microaneurysms or a dot/blot hemorrhage, but this describes model attention only — it is not a confirmed clinical finding.`
      : `diffuse, spread across a broader area (top 10% of pixels account for only ${(desc.concentration*100).toFixed(0)}% of total activation). Diffuse attention like this does not by itself localize a specific lesion.`;

    const quadrantSource = anatomy
      ? `The quadrant is named relative to the estimated optic disc position shown in 3b, so nasal and temporal are anatomically meaningful rather than just left and right of the frame.`
      : `Landmark estimation failed on this image, so the region is described in plain frame terms rather than by retinal quadrant.`;

    const countText = findings && findings.length
      ? `${findings.length} region${findings.length === 1 ? " is" : "s are"} marked on the gradient above.`
      : (desc.flat
          ? `No region is marked. The map is warm across the whole retina rather than peaked anywhere in it, so there is no hotspot to circle. Grad-CAM is always scaled to its own maximum, which means something always reaches full intensity — marking it here would invent a localisation the model did not make.`
          : `No region is marked: nothing rose far enough above the retina's own background activation to count as a hotspot.`);

    const fieldNote = `Activation is measured over the retina only, inside the dashed circle. The map is masked to the fitted camera aperture, trimmed by half a Grad-CAM cell, and rescaled inside that mask — so the black surround can neither be marked as a finding nor set the scale everything else is judged against. The rim outside the dashed circle is real retina that is deliberately not judged: at this layer's resolution a cell straddling the aperture edge is computed mostly from the black surround, so its value says nothing about retina. Nothing there is assessed, and nothing there is ruled out.`;

    note.innerHTML = `Peak Grad-CAM activation falls in the <strong>${desc.region}</strong> quadrant. ${countText} Attention is ${focalText} ${quadrantSource} ${fieldNote}`;
  }

  // ---------------------------------------------------------------------
  // RENDER: SUMMARY REPORT
  // ---------------------------------------------------------------------
  function renderReport(){
    const q = state.lastQuality, g = state.lastGrading, gc = state.lastGradcamDesc;
    const body = $("#report-body");
    let html = `<p class="small">Generated ${new Date().toLocaleString()}</p>`;

    html += `<h3>Image quality</h3><table class="kv">
      <tr><td>Focus (Laplacian variance)</td><td>${q.metrics.focusVar.toFixed(1)}</td></tr>
      <tr><td>Brightness mean</td><td>${q.metrics.brightMean.toFixed(1)} / 255</td></tr>
      <tr><td>Field of view</td><td>${(q.metrics.fov*100).toFixed(0)}%</td></tr>
      <tr><td>Verdict</td><td>${q.verdict.verdict === "reject" ? "Rejected — " + q.verdict.reason : (q.verdict.verdict === "enhance" ? "Enhanced then screened" : "Screened without enhancement")}</td></tr>
    </table>`;

    if (g){
      html += `<h3>DR classification</h3><table class="kv">
        <tr><td>${CLASS_NAMES[0]}</td><td>${(g.probs[0]*100).toFixed(1)}%</td></tr>
        <tr><td>${CLASS_NAMES[1]}</td><td>${(g.probs[1]*100).toFixed(1)}%</td></tr>
        <tr><td>Predicted class</td><td>${CLASS_NAMES[g.predClass]}</td></tr>
        <tr><td>Referable verdict</td><td>${g.predClass===1 ? "Referable — refer to ophthalmologist for full ICDR grading" : "Non-referable"}</td></tr>
      </table>`;
    }
    if (gc){
      html += `<h3>Grad-CAM finding</h3><p>Peak attention region: <strong>${gc.region}</strong>. Attention pattern: ${gc.focal ? "focal" : "diffuse"} (${(gc.concentration*100).toFixed(0)}% of activation in top 10% of pixels).</p>`;
      const findings = state.lastFindings;
      if (findings && findings.length){
        html += `<table class="kv">` + findings.map(f =>
          `<tr><td>Marked region ${f.index}</td><td>${f.quadrant} quadrant · peak ${(f.peak*100).toFixed(0)}% of maximum · ${f.areaPct.toFixed(1)}% of frame</td></tr>`
        ).join("") + `</table>`;
      } else {
        html += `<p class="small">${gc.flat
          ? "No region marked — activation is spread across the whole retina rather than peaked anywhere in it, so there is no hotspot to localise."
          : "No region marked — nothing rose far enough above the retina's own background activation to count as a hotspot."} Activation is measured over the retina only; the black surround outside the camera aperture is excluded and cannot be marked.</p>`;
      }
      if (state.lastAnatomy){
        const a = state.lastAnatomy;
        html += `<p class="small">Quadrants are named relative to the estimated optic disc at x ${Math.round(a.disc.x)}, y ${Math.round(a.disc.y)} px, with the nasal side to the ${a.nasalSide} of that axis. Landmark positions are heuristic estimates for orientation, not a trained detection.</p>`;
      }
    }
    const sev = state.lastSeverity;
    if (sev){
      html += `<h3>ICDR severity estimate</h3><table class="kv">
        <tr><td>Grade</td><td>${sev.level === null ? "not assessed" : "Level " + sev.level + " — " + sev.label}</td></tr>
        <tr><td>Referable</td><td>${sev.referable ? "yes, moderate NPDR or worse under the rule" : "not by this estimate"}</td></tr>
        <tr><td>Specialist review</td><td>${sev.refer ? "indicated — " + sev.reason : "not flagged by this estimate"}</td></tr>
      </table>`;
      html += `<p class="small">Applied to unvalidated candidates, so the grade inherits their errors. Venous beading, IRMA and neovascularisation are not detectable by this method, so severe and proliferative disease can never be excluded here.</p>`;
      if (sev.doubts.length){
        html += `<p class="small"><strong>Doubts on this image:</strong> ${sev.doubts.join("; ")}.</p>`;
      }
    }

    const les = state.lastLesions;
    if (les && les.candidates.length){
      html += `<h3>Lesion candidates (classical morphology, not the CNN)</h3><table class="kv">` +
        LESION_ORDER.map(k => `<tr><td>${LESION_TYPES[k].label}</td><td>${les.counts[k]}</td></tr>`).join("") +
        `<tr><td>Total candidate regions</td><td>${les.candidates.length}</td></tr></table>` +
        `<p class="small">Unvalidated candidates from morphological filtering, independent of the classifier and of each other. Not a segmentation result and not a lesion count for grading.</p>`;
    } else if (les){
      html += `<h3>Lesion candidates</h3><p>No candidate regions passed the filters. This does not indicate a normal retina.</p>`;
    } else {
      html += `<h3>Lesion candidates</h3><p>Not computed for this image.</p>`;
    }
    html += `<p class="small">Research/hackathon prototype for SIH26038. Not a certified medical device. Classifier provenance and limitations are documented in the page footer.</p>`;
    body.innerHTML = html;
  }

  // ---------------------------------------------------------------------
  // MAIN PIPELINE
  // ---------------------------------------------------------------------

  // Clear every per-image output before a new run, so a failed stage on image 2
  // can never leave image 1's overlay or text on screen.
  function resetImageOutputs(){
    ["canvas-enhanced", "canvas-gradcam", "canvas-gradcam-colour", "canvas-anatomy",
     "canvas-mask", "canvas-mask-typed", "canvas-lesion-overlay"].forEach(id => {
      const c = document.getElementById(id);
      if (!c) return;
      c.getContext("2d").clearRect(0, 0, c.width, c.height);
      c.width = 1; c.height = 1;
    });
    ["findings-legend", "anatomy-table", "gradcam-note", "grading-results", "quality-notice",
     "lesion-legend", "lesion-summary", "lesion-quadrants"].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.innerHTML = "";
    });
    state.lastGrading = null;
    state.lastGradcamDesc = null;
    state.lastCam = null;
    state.lastAnatomy = null;
    state.lastFindings = null;
    state.lastLesions = null;
    state.lastSeverity = null;
    const rb = document.getElementById("refer-banner");
    if (rb) rb.classList.remove("visible");
    ["grade-result","grade-quadrants"].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.innerHTML = "";
    });
  }

  async function runPipeline(file){
    $("#run-button").disabled = true;
    resetImageOutputs();
    try{
      const img = await loadImageFile(file);
      const working = document.getElementById("canvas-original");
      drawToCanvas(img, working, WORKING_MAX_DIM);
      state.workingCanvas = working;

      // STEP 1: quality
      const metrics = computeMetrics(working);
      const verdict = assessQuality(metrics);
      state.lastQuality = { metrics, verdict };
      showStep("step1");
      renderQualityStep(metrics, verdict);

      const enhancedCanvas = document.getElementById("canvas-enhanced");
      if (verdict.verdict === "reject"){
        enhancedCanvas.getContext("2d").clearRect(0,0,enhancedCanvas.width,enhancedCanvas.height);
        enhancedCanvas.width = 1; enhancedCanvas.height = 1;
        // Nothing was assessed, so nothing can be reassuring. Screening an image
        // this poor would produce a confident-looking result with no basis.
        referWithoutGrade(
          "This image could not be assessed at all, so nothing here rules anything out.",
          ["the image failed the quality check: " + (verdict.reason || "quality too low to analyse"),
           "recapture if you can, and refer if a usable image cannot be obtained"]);
        $("#run-button").disabled = false;
        return; // do not proceed to inference
      }

      const procCanvas = copyCanvas(working);
      if (verdict.verdict === "enhance"){
        enhanceImage(procCanvas);
      }
      state.procCanvas = procCanvas;   // kept so detection can be re-run alone
      enhancedCanvas.width = procCanvas.width;
      enhancedCanvas.height = procCanvas.height;
      enhancedCanvas.getContext("2d").drawImage(procCanvas, 0, 0);

      // The classifier and Grad-CAM need the model. Landmarks, lesion detection
      // and the grading rule do not, so a failed model download costs those
      // stages nothing and they still run.
      const modelRan = state.modelReady;

      showStep("step2");
      if (!modelRan){
        $("#model-scope-notice").innerHTML = `<strong>Classifier unavailable — demo mode.</strong> The TensorFlow.js model failed to load in this session, so no classification and no Grad-CAM are shown. This is disclosed rather than substituted with placeholder numbers. The lesion detector below is classical image processing and does not need the model, so it still runs — but nothing cross-checks it.`;
        $("#grading-results").innerHTML = "";
      } else {
        // STEP 2: grading
        const gradingResult = await runInference(procCanvas);
        state.lastGrading = gradingResult;
        renderGradingStep(gradingResult);

      }

      // 3b: anatomical landmarks. Computed before Grad-CAM because both step 3
      // (quadrant naming) and step 4 (masking out the optic disc) depend on them.
      let anatomy = null;
      try{
        const disc = estimateOpticDisc(procCanvas);
        const fm = estimateFoveaMacula(procCanvas, disc);
        anatomy = { disc, fovea: fm.fovea, maculaRadius: fm.maculaRadius, nasalSide: fm.nasalSide, dir: fm.dir, evidence: fm.evidence };
        drawAnatomyOverlay(procCanvas, anatomy, document.getElementById("canvas-anatomy"));
        renderAnatomyTable(anatomy);
      } catch(anErr){
        console.error("Landmark estimation failed", anErr);
        document.getElementById("anatomy-table").innerHTML =
          `<tr><td>Landmarks</td><td>Estimation failed on this image (${anErr.message||anErr}) — no landmarks are drawn rather than guessed positions being shown.</td></tr>`;
      }
      state.lastAnatomy = anatomy;
      showStep("step3");

      // STEP 3: Grad-CAM (class index 1 = "DR present", regardless of predicted class)
      try{
        if (!modelRan) throw new Error("classifier unavailable, so there are no gradients to read");
        const cam = await computeGradCAM(procCanvas, 1);

        // 3a: gradient overlay, then circle the connected high-activation regions
        const desc = describeGradCAM(cam, anatomy);
        state.lastGradcamDesc = desc;
        state.lastCam = cam;
        const marked = redrawCamRegions(procCanvas, cam, anatomy, desc);
        renderGradCAMStep(desc, marked.findings, anatomy);
      } catch(gcErr){
        console.error("Grad-CAM failed", gcErr);
        $("#gradcam-note").innerHTML = `Grad-CAM computation failed in this session (${gcErr.message||gcErr}). No heatmap is shown — this is disclosed rather than faked.`;
      }

      // STEP 4: lesion candidate detection (classical morphology, independent of the CNN)
      showStep("step4");
      document.getElementById("lesion-summary").innerHTML =
        '<p class="notice">Scanning for lesion candidates. This takes a few seconds — the directional morphology runs at full resolution because halving it was measured to be far less accurate.</p>';
      await yieldToBrowser();
      try{
        const lesions = detectLesionCandidates(procCanvas, anatomy);
        state.lastLesions = lesions;
        renderBinaryMask(lesions, document.getElementById("canvas-mask"));
        renderTypedMask(lesions, document.getElementById("canvas-mask-typed"));
        renderLesionOverlay(procCanvas, lesions, document.getElementById("canvas-lesion-overlay"));
        renderLesionTables(lesions, anatomy);

        // Step 5: apply the published grading rule to what was found.
        const sev = assessSeverity(lesions, anatomy, state.lastGrading, state.lastQuality);
        state.lastSeverity = sev;
        showStep("severity");
        renderSeverity(sev);
        renderReferBanner(sev);
      } catch(lesErr){
        console.error("Lesion candidate detection failed", lesErr);
        document.getElementById("lesion-summary").innerHTML =
          `<p class="notice">Lesion candidate detection failed on this image (${lesErr.message||lesErr}). No mask is shown rather than an empty one being passed off as a clear retina.</p>`;
        const sev = assessSeverity(null, anatomy, state.lastGrading, state.lastQuality);
        state.lastSeverity = sev;
        showStep("severity");
        renderSeverity(sev);
        renderReferBanner(sev);
      }

      // SUMMARY
      showStep("report");
      renderReport();

    } catch(err){
      console.error(err);
      alert("Could not process this image: " + (err.message || err));
    } finally{
      $("#run-button").disabled = false;
    }
  }

  // ---------------------------------------------------------------------
  // WIRING
  // ---------------------------------------------------------------------
  window.addEventListener("DOMContentLoaded", () => {
    loadModel();

    const thresholdText = document.getElementById("blob-threshold-text");
    if (thresholdText) thresholdText.textContent = Math.round(BLOB_THRESHOLD*100) + "%";

    $("#file-input").addEventListener("change", () => {
      $("#run-button").disabled = !state.modelLoadAttempted || $("#file-input").files.length === 0;
    });

    $("#run-button").addEventListener("click", () => {
      const files = $("#file-input").files;
      if (files.length === 0) return;
      runPipeline(files[0]);
    });

    $("#rerun-cam").addEventListener("click", () => {
      if (!state.lastCam || !state.procCanvas){
        $("#gradcam-note").textContent = "Run the pipeline on an image first.";
        return;
      }
      BLOB_THRESHOLD = clamp(parseFloat($("#tune-cam-threshold").value) || BLOB_THRESHOLD, 0.15, 0.95);
      CAM_REGION_LIMIT = Math.max(1, Math.round(parseFloat($("#tune-cam-limit").value) || CAM_REGION_LIMIT));
      const marked = redrawCamRegions(state.procCanvas, state.lastCam, state.lastAnatomy, state.lastGradcamDesc);
      if (state.lastGradcamDesc) renderGradCAMStep(state.lastGradcamDesc, marked.findings, state.lastAnatomy);
      if (state.lastQuality) renderReport();
    });

    $("#rerun-detection").addEventListener("click", async () => {
      if (!state.procCanvas){
        document.getElementById("lesion-summary").innerHTML =
          '<p class="notice">Run the pipeline on an image first.</p>';
        return;
      }
      SEED_FLOOR_DARK = parseFloat($("#tune-dark").value) || SEED_FLOOR_DARK;
      SEED_FLOOR_BRIGHT = parseFloat($("#tune-bright").value) || SEED_FLOOR_BRIGHT;
      MIN_CONTRAST_K = parseFloat($("#tune-contrast").value) || MIN_CONTRAST_K;
      // Below 1 nothing can ever pass, which would silently empty the overlay.
      SATURATION_MAX_GROWTH = clamp(parseFloat($("#tune-growth").value) || SATURATION_MAX_GROWTH, 1.0, 3.0);
      const btn = $("#rerun-detection");
      btn.disabled = true;
      document.getElementById("lesion-summary").innerHTML =
        '<p class="notice">Re-running detection at the new sensitivity…</p>';
      await yieldToBrowser();
      try{
        const lesions = detectLesionCandidates(state.procCanvas, state.lastAnatomy);
        state.lastLesions = lesions;
        renderBinaryMask(lesions, document.getElementById("canvas-mask"));
        renderTypedMask(lesions, document.getElementById("canvas-mask-typed"));
        renderLesionOverlay(state.procCanvas, lesions, document.getElementById("canvas-lesion-overlay"));
        renderLesionTables(lesions, state.lastAnatomy);
        const sev = assessSeverity(lesions, state.lastAnatomy, state.lastGrading, state.lastQuality);
        state.lastSeverity = sev;
        showStep("severity");
        renderSeverity(sev);
        renderReferBanner(sev);
        if (state.lastQuality) renderReport();
      } catch(err){
        console.error(err);
        document.getElementById("lesion-summary").innerHTML =
          '<p class="notice">Detection failed at these settings (' + (err.message||err) + ').</p>';
      } finally{
        btn.disabled = false;
      }
    });

    $("#print-button").addEventListener("click", () => {
      document.querySelectorAll("details").forEach(d => d.open = true);
      window.print();
    });
  });

})();
