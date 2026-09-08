function R = runPipeline(input, opts)
%RUNPIPELINE  The whole DR screening pipeline, Step 1 through Step 5.
%
%   R = dr.runPipeline(FILENAME) reads and analyses one fundus image.
%   R = dr.runPipeline(I) analyses an image already in memory.
%   R = dr.runPipeline(..., Net=NET, Render=true) supplies a classifier and asks
%   for the annotated panels.
%
%   Returns a struct carrying every stage's output, including the stages that
%   did not run and why.
%
%   ---- The structure, and why it is this way ----
%   Steps 1, 3, 4 and 5 are classical image processing. Step 2 (classification)
%   and Step 3a (Grad-CAM) need the network. They are deliberately decoupled: a
%   missing or failed model costs the classifier and the heatmap and nothing
%   else, so a district deployment with no model file still produces a lesion map
%   and an ICDR estimate. What it does not produce is the cross-check between the
%   two, and DR.ASSESSSEVERITY records that absence as a reason for caution
%   rather than letting it pass silently.
%
%   The quality gate is a hard stop, not a warning. An image that cannot be
%   assessed produces no grade at all, because a confident-looking result with
%   nothing underneath it is the one that gets acted on in the field.

    arguments
        input
        opts.Net = []
        opts.Render (1,1) logical = false
        opts.Verbose (1,1) logical = true
    end

    R = struct();
    R.timestamp = datetime("now");

    % ---- Load -----------------------------------------------------------
    if isstring(input) || ischar(input)
        [I, meta] = dr.readFundus(input);
        R.meta = meta;
    else
        I = im2uint8(input);
        C = dr.Config();
        s = min(1, C.workingMaxDim / max(size(I,1), size(I,2)));
        if s < 1
            I = imresize(I, s, "bilinear");
        end
        R.meta = struct("source", "array", "workingSize", [size(I,1) size(I,2)]);
    end
    R.working = I;

    % ---- STEP 1: quality --------------------------------------------------
    R.metrics = dr.computeMetrics(I);
    R.quality = dr.assessQuality(R.metrics);
    say(opts.Verbose, "Step 1  quality: %s", R.quality.verdict);

    if R.quality.verdict == "reject"
        R.stopped = true;
        R.severity = dr.assessSeverity([], [], [], R.quality);
        R.severity.reason = "This image could not be assessed at all, so nothing " + ...
                            "here rules anything out. " + R.quality.reason;
        say(opts.Verbose, "  rejected: %s", R.quality.reason);
        return;
    end
    R.stopped = false;

    % ---- STEP 1b: enhancement --------------------------------------------
    if R.quality.verdict == "enhance"
        proc = dr.enhanceImage(I);
        R.enhanced = true;
    else
        proc = I;
        R.enhanced = false;
    end
    R.processed = proc;

    % ---- STEP 3 (first): anatomical landmarks -----------------------------
    % Computed before Grad-CAM because both the quadrant naming in Step 3 and the
    % disc exclusion in Step 4 depend on them.
    anatomy = [];
    try
        disc = dr.estimateOpticDisc(proc);
        fm = dr.estimateFoveaMacula(proc, disc);
        anatomy = struct("disc", disc, "fovea", fm.fovea, ...
                         "maculaRadius", fm.maculaRadius, ...
                         "nasalSide", fm.nasalSide, "evidence", fm.evidence);
        say(opts.Verbose, "Step 3  disc r=%.1f px (%s), fovea at %.0f,%.0f", ...
            disc.radius, ternary(disc.measuredExtent, "grown", "plateau"), ...
            fm.fovea.x, fm.fovea.y);
    catch err
        R.anatomyError = err.message;
        say(opts.Verbose, "Step 3  landmark estimation failed: %s", err.message);
    end
    R.anatomy = anatomy;

    % ---- STEP 2: classification -------------------------------------------
    net = opts.Net;
    if isempty(net)
        net = dr.loadClassifier();
    end
    R.modelAvailable = ~isempty(net);

    if R.modelAvailable
        R.grading = dr.classifyImage(net, proc);
        say(opts.Verbose, "Step 2  P(DR) = %.3f in %.0f ms", ...
            R.grading.probs(2), R.grading.elapsedMs);
    else
        R.grading = [];
        say(opts.Verbose, "Step 2  skipped - no classifier available");
    end

    % ---- STEP 3a: Grad-CAM ------------------------------------------------
    % Always for the "DR present" class, whatever was predicted: what a clinician
    % needs is where the evidence for disease would have been.
    if R.modelAvailable
        try
            R.cam = dr.computeGradCAM(net, proc, 2);
            R.camDescription = dr.describeGradCAM(R.cam, anatomy);
            say(opts.Verbose, "Step 3a Grad-CAM peak in %s, concentration %.2f", ...
                R.camDescription.region, R.camDescription.concentration);
        catch err
            R.camError = err.message;
            say(opts.Verbose, "Step 3a Grad-CAM failed: %s", err.message);
        end
    end

    % ---- STEP 4: lesion candidates ----------------------------------------
    try
        R.lesions = dr.detectLesionCandidates(proc, anatomy);
        say(opts.Verbose, "Step 4  %d candidates (MA %d, HEM %d, HE %d, CWS %d)", ...
            numel(R.lesions.candidates), R.lesions.counts.MA, R.lesions.counts.HEM, ...
            R.lesions.counts.HE, R.lesions.counts.CWS);
    catch err
        R.lesions = [];
        R.lesionError = err.message;
        say(opts.Verbose, "Step 4  lesion detection failed: %s", err.message);
    end

    % ---- STEP 5: ICDR severity --------------------------------------------
    R.severity = dr.assessSeverity(R.lesions, anatomy, R.grading, R.quality);
    say(opts.Verbose, "Step 5  level %s - %s (refer: %d)", ...
        string(R.severity.level), R.severity.label, R.severity.refer);

    % ---- Panels -----------------------------------------------------------
    if opts.Render && ~isempty(R.lesions)
        if isfield(R, "cam")
            R.panels = dr.renderOverlays(proc, anatomy, R.lesions, R.cam);
        else
            R.panels = dr.renderOverlays(proc, anatomy, R.lesions);
        end
    end
end

function say(verbose, fmt, varargin)
    if verbose
        fprintf(fmt + "\n", varargin{:});
    end
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
