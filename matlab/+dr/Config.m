function C = Config()
%CONFIG  Every tuned constant in the DR screening pipeline, in one place.
%
%   C = dr.Config() returns a struct of the thresholds, scales and limits the
%   pipeline runs on. They are gathered here rather than scattered through the
%   functions because almost every one of them was arrived at by measurement,
%   and a number you cannot find is a number nobody can re-measure.
%
%   The values are carried across unchanged from the reference implementation in
%   js/app.js. Where a constant encodes a decision rather than a scale, the
%   reasoning is kept with it — the reasoning is the part that is hard to
%   reconstruct later.
%
%   See also DR.RUNPIPELINE, DR.DETECTLESIONCANDIDATES.

% ---- Working sizes ------------------------------------------------------
C.workingMaxDim = 800;      % cap on the working image's long edge
C.metricDim     = 512;      % standardised size for quality metrics, so the
                            % thresholds below are resolution-independent

% ---- Step 1: quality gate ----------------------------------------------
C.quality.minFocusVar      = 15;
C.quality.enhanceFocusVar  = 40;
C.quality.minBrightMean    = 40;
C.quality.maxBrightMean    = 235;
C.quality.enhanceBrightLow = 60;
C.quality.enhanceBrightHigh= 200;
C.quality.minFov           = 0.35;
C.quality.enhanceFov       = 0.5;

% ---- Step 1b: enhancement ----------------------------------------------
C.enhance.tiles      = [8 8];
% adapthisteq takes a *normalised* clip limit, the reference implementation a
% multiple of the tile's mean bin count. The two are related by
%   clip = P/B + L*(P - P/B)      (MATLAB, P pixels per tile, B bins)
%   clip = k*P/B                  (reference, k = 3.5)
% so L = (k-1)/(B-1) = 2.5/255. Derived rather than guessed, so the contrast
% this stage produces matches the reference rather than merely resembling it.
C.enhance.clipLimit  = 2.5/255;
C.enhance.numBins    = 256;
C.enhance.denoiseMix = 0.3;      % how much of a radius-1 box blur to blend in

% ---- Step 2: classifier -------------------------------------------------
C.model.inputSize   = [224 224];
C.model.classNames  = ["No DR", "DR present"];
C.model.featureLayerCandidates = ["conv_pw_11_relu", "conv_pw_13_relu"];

% ---- Step 3: Grad-CAM ---------------------------------------------------
C.cam.blobThreshold  = 0.55;   % fraction of peak activation a region must reach
C.cam.minAreaFrac    = 0.004;  % and how much of the frame it must cover
C.cam.regionLimit    = 8;
% Grad-CAM is normalised to its own maximum, so *something* always reaches 1.0
% and a fraction-of-peak cut alone always returns a region — on a healthy retina,
% the tallest bump in a flat map. A region must also stand this many robust
% deviations clear of the retina's own typical activation.
C.cam.noiseK         = 3.0;
% ...and a map whose typical retinal pixel is already this warm localises
% nothing, whatever its peak, so no region is marked at all.
C.cam.flatMedian     = 0.55;
C.cam.gridHint       = 14;      % CAM cells across the frame, for the edge trim
C.cam.featherFrac    = 0.7;

% ---- Step 4: lesion detection ------------------------------------------
% Deviation thresholds, as multiples of the image's own robust noise scale plus
% an absolute floor. Both are needed: the multiple adapts to the camera, the
% floor stops a very clean image from dropping its threshold into the texture.
C.lesion.darkK   = 3.5;
C.lesion.brightK = 4.0;
C.lesion.darkFloor   = 10;
C.lesion.brightFloor = 12;
C.lesion.darkContrastFloor   = 9;
C.lesion.brightContrastFloor = 11;
C.lesion.minContrastK = 3.0;

C.lesion.cwsMinAreaFrac = 0.0004;   % cotton wool spots are a quarter disc across
C.lesion.edgeZoneFrac      = 0.80;  % beyond this fraction of the aperture radius
C.lesion.edgeArtifactAreaFrac = 0.0015;  % a region this large is a rim artifact
C.lesion.hystRatio = 0.45;          % permissive threshold, as a fraction of strict

% Vessel map. A vessel is told from a lesion by the *spread* of the directional
% closing across orientations, not by the size of any one response: at a vessel
% the element lying along it fills nothing while every crossing element fills it
% completely, so the responses are far apart. At a round lesion they are close.
C.lesion.vesselElongation = 1.5;
C.lesion.vesselSeedQ = 0.95;
C.lesion.vesselGrowQ = 0.80;
C.lesion.vesselOverlapReject = 0.78;

% Linear structuring element, as a fraction of the short edge. Too short and
% thin vessels are filled and read as lesions; too long and short vessel
% segments start to read as lesions.
C.lesion.linearSEFrac = 0.014;
C.lesion.linearOrientations = 12;
C.lesion.seedTailQ = 0.97;
C.lesion.seedFloorDark = 6;
C.lesion.seedFloorBright = 8;

% Scale saturation. A bounded blob is already filled at the element matching its
% own width, so the next element up adds nothing; anything that continues past
% the element keeps gaining at every rung. That, not amplitude, is what separates
% a lesion from a vessel bend or a patch of choroidal mottling.
C.lesion.saturationSEMult = 2.0;
C.lesion.saturationLadder = 3;      % seL, 2*seL, 4*seL
C.lesion.saturationOrientations = 6;
C.lesion.saturationMaxGrowth = 1.10;

% Vessel junction test. Where two vessels cross, where one branches and at the
% apex of a tight bend there is no direction in which the structure is straight,
% so every orientation fills it and it answers the roundness test exactly as a
% lesion does. What differs is what runs out of it.
C.lesion.armDirections = 24;
C.lesion.armPersistence = 0.55;
C.lesion.armCollarPad = 3;
C.lesion.junctionAntipodalTolDeg = 45;
C.lesion.junctionRingSaturation = 0.70;
C.lesion.onVesselContrastFactor = 1.35;

% Broad smooth darkening is anatomy or shadow, not a lesion.
C.lesion.smoothGradientRatio = 0.55;
C.lesion.maculaSmoothAreaFrac = 0.08;
C.lesion.smoothBroadAreaFrac = 0.004;

% Above this many candidates the detector is almost certainly responding to
% image noise, and the report says so instead of presenting a tidy count.
C.lesion.noiseSuspicionCount = 400;

% ---- Peripapillary bright suppression ----------------------------------
% The optic disc is the brightest thing in a normal fundus, so every bright
% candidate it produces is a false exudate. Excluding a circle of 1.15 disc radii
% only works as well as the radius does, and a radius read off a plateau at a
% fixed height above the *global* retinal mean lands inside the rim. The rim, a
% scleral crescent and the peripapillary reflex then fall in the gap between
% where the circle stops and where the disc ends — and each is bright, round and
% sharply bounded, which is the description of a hard exudate.
C.disc.peripapillaryMult = 2.5;
C.disc.edgeLevel = 0.50;     % half-maximum: where a blurred edge actually sits
C.disc.haloLevel = 0.28;     % lower, to take in crescent and reflex
C.disc.fillMaxMult = 3.2;    % past this the fill has escaped into open retina
C.disc.tissueOverlap = 0.33; % of a bright region, before the disc explains it
C.disc.colourLevel = 0.55;   % on the retina(0)->disc(1) blue-fraction scale
C.disc.colourMinSeparation = 0.02;   % below this the reference is degenerate

% ---- Step 5: ICDR grading ----------------------------------------------
% The "4" arm of the 4-2-1 rule for severe NPDR: more than twenty intraretinal
% hemorrhages in each of the four quadrants.
C.icdr.hemPerQuadrant = 20;
C.icdr.quadrantNames = ["Superior-Nasal","Superior-Temporal", ...
                        "Inferior-Nasal","Inferior-Temporal"];

% ---- Lesion classes -----------------------------------------------------
C.lesionOrder = ["MA","HEM","HE","CWS"];
C.lesionTypes.MA  = struct("key","MA", "label","Microaneurysm candidate",       "grey",255);
C.lesionTypes.HEM = struct("key","HEM","label","Dot/blot hemorrhage candidate", "grey",200);
C.lesionTypes.HE  = struct("key","HE", "label","Hard exudate candidate",        "grey",150);
C.lesionTypes.CWS = struct("key","CWS","label","Cotton wool spot candidate",    "grey",100);
end
