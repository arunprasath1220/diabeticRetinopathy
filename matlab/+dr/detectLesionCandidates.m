function out = detectLesionCandidates(I, anatomy)
%DETECTLESIONCANDIDATES  Step 4: find and classify lesion candidates.
%
%   OUT = dr.detectLesionCandidates(I, ANATOMY) returns a struct with the
%   candidate list, per-type counts, the rejection tally and the measurement
%   context needed to report on any of it.
%
%   Classical morphology throughout, entirely independent of the CNN. That
%   independence is the point: Step 2 and Step 4 reach their conclusions by
%   different means, so when they disagree the disagreement is informative, and
%   DR.ASSESSSEVERITY treats it as a reason to refer.
%
%   The method:
%     1. Mask and erode the retinal field, so the field-of-view edge - a huge
%        intensity step - cannot generate candidates.
%     2. Seed from directional top-hats (DR.ROUNDSTRUCTURESEEDS). No directional
%        response anywhere in a region means no lesion, and that single test
%        excludes the vessels, the macula and smooth background shading at once,
%        without needing to know where any of them are.
%     3. Grow each seed into its full extent against a multi-scale background.
%     4. Reject on measured shape, boundedness, position and contrast.
%     5. Type what survives by size, shape and edge sharpness.
%
%   ANATOMY may be [], in which case the disc and macula rules are skipped and
%   the caller is told so through OUT.discExcluded / OUT.maculaExcluded.

    C = dr.Config();
    L = C.lesion;

    ch = dr.imageChannels(I);
    w = ch.w;  h = ch.h;
    green = ch.green;  lum = ch.lum;  red = ch.red;  blue = ch.blue;

    inside = dr.retinaFieldMask(lum, 0.05);
    retinaArea = nnz(inside);
    if retinaArea < numel(lum)*0.05
        error("dr:detectLesionCandidates:tinyField", ...
              "retinal area too small to analyse");
    end

    greenS = dr.boxBlur(green, 1);
    lumS   = dr.boxBlur(lum, 1);

    % Edge strength, central differences. Hard exudates have sharp margins and
    % cotton wool spots do not, which is the only thing that separates them.
    grad = zeros(h, w);
    if h > 2 && w > 2
        gx = lumS(2:end-1, 3:end)   - lumS(2:end-1, 1:end-2);
        gy = lumS(3:end,   2:end-1) - lumS(1:end-2, 2:end-1);
        grad(2:end-1, 2:end-1) = hypot(gx, gy);
    end

    % ---- Lesion evidence ------------------------------------------------
    seeds = dr.roundStructureSeeds(greenS, lumS, inside);
    darkSeed   = seeds.darkSeed;
    brightSeed = seeds.brightSeed;
    vesselMask = seeds.vesselMask;
    vesselZone = dr.dilateMask(vesselMask, 1);

    [X, Y] = meshgrid(1:w, 1:h);

    % ---- Scale saturation ------------------------------------------------
    % Widest structure the ladder can still speak for. Past this the longest
    % element no longer fills the region, so no rung can show saturation, and
    % silence must not be read as a verdict - this is what keeps a genuinely
    % large blot hemorrhage from being deleted by a rule that was never about it.
    satMaxArea = pi * seeds.satSE(end)^2;

    % ---- Backgrounds ------------------------------------------------------
    % Vessels are excluded from the averaging as well as the black surround. A
    % vessel in the window drags the background down and makes ordinary retina
    % between two vessels read as abnormally bright.
    bgMask = inside & ~vesselZone;

    % ---- Extent. Seeds say where lesions are; these say how far they reach.
    scales = [ max(4,  round(min(w,h)*0.015)), ...
               max(10, round(min(w,h)*0.055)), ...
               max(20, round(min(w,h)*0.120)) ];

    darkLoose   = false(h, w);   brightLoose = false(h, w);
    darkDev     = zeros(h, w);   brightDev   = zeros(h, w);
    noiseDark = 1;  noiseBright = 1;

    for si = 1:numel(scales)
        r = scales(si);
        bgG = dr.maskedBlur(greenS, bgMask, r);
        bgL = dr.maskedBlur(lumS,   bgMask, r);

        dDark   = bgG - greenS;      % dark lesions are dark in green
        dBright = lumS - bgL;        % bright lesions are bright in luminance

        sDark   = dr.robustSigma(dDark,   inside);
        sBright = dr.robustSigma(dBright, inside);
        if si == 1
            noiseDark = sDark;  noiseBright = sBright;
        end

        tDark   = max(L.darkFloor,   sDark  *L.darkK)   * L.hystRatio;
        tBright = max(L.brightFloor, sBright*L.brightK) * L.hystRatio;

        darkLoose   = darkLoose   | (inside & (dDark   > tDark));
        brightLoose = brightLoose | (inside & (dBright > tBright));
        darkDev   = max(darkDev,   dDark);
        brightDev = max(brightDev, dBright);
    end

    % Growth must not run along a vessel. A lesion lying on one is seeded
    % correctly, but the permissive mask covers the vessel as well, so the region
    % grows out along the whole vessel tree and is then thrown away as too large
    % or too elongated - taking the lesion with it. Vessel pixels are therefore
    % removed from what a region may grow into.
    darkLoose(vesselMask)   = false;
    brightLoose(vesselMask) = false;

    % ...and then every pixel that still carries lesion evidence is put back.
    % Restoring only the seeds is not enough once the vessel map is dense: a
    % lesion lying against a vessel has its seed restored but the rest of its body
    % deleted, so what is left to grow from is the sliver furthest from the vessel
    % - the lesion's own rim, which is the one shape that reads as unbounded, and
    % it is then thrown out by the saturation test. This cannot re-admit the
    % vessel, because it is the roundness response and a vessel barely produces
    % one: that is the whole basis on which the vessel map was built.
    darkKeep   = seeds.tDark   * L.hystRatio;
    brightKeep = seeds.tBright * L.hystRatio;
    darkLoose   = darkLoose   | darkSeed   | (seeds.roundDark   > darkKeep);
    brightLoose = brightLoose | brightSeed | (seeds.roundBright > brightKeep);
    darkLoose(~inside)   = false;
    brightLoose(~inside) = false;

    meanGrad = mean(grad(inside));
    sharpEdgeThresh = meanGrad * 1.5;

    n = numel(lum);
    maxArea   = round(n*0.030);
    minArea   = max(6, round(n*0.000012));
    maMaxArea = max(minArea+1, round(n*0.00018));
    cwsMinArea = round(n*L.cwsMinAreaFrac);

    % ---- Position rules ---------------------------------------------------
    hasAnatomy = ~isempty(anatomy);
    if hasAnatomy
        discR = anatomy.disc.radius;
        discD2 = (X - anatomy.disc.x).^2 + (Y - anatomy.disc.y).^2;
    else
        discR = 0;
        discD2 = inf(h, w);
    end
    hasMacula = hasAnatomy && isfield(anatomy, "fovea") && ~isempty(anatomy.fovea);

    % Measured disc tissue. The size is checked rather than assumed: landmarks
    % and detection run on the same image today, but a mask indexed against a
    % different size would not fail loudly - it would quietly delete findings
    % somewhere else in the image, which is the worst way for this to go wrong.
    discTissue = [];
    if hasAnatomy && isfield(anatomy.disc, "tissue") && ~isempty(anatomy.disc.tissue)
        t = anatomy.disc.tissue;
        if isequal(size(t), [h w])
            discTissue = logical(t);
        end
    end

    % ---- Colour reference for the peripapillary ring ---------------------
    % Past the halo there is still nerve-fibre reflex and atrophic mottling, and
    % in shape, size and contrast those are indistinguishable from a hard exudate
    % - which is why every geometric rule tried here left some of them standing.
    % What separates them is what they are made of. An exudate is lipid and reads
    % yellow: it gains in red and green and hardly at all in blue. Disc tissue,
    % sclera and reflex are white or grey and gain in all three, so their blue
    % fraction climbs while an exudate's stays near the retina's.
    %
    % Both ends of the scale are read off this image, so nothing here depends on
    % the camera's white balance, and an image where the two ends do not separate
    % gets no colour test rather than a guessed one.
    chanSum = red + green + blue;
    blueFrac = -ones(h, w);
    lit = chanSum > 24;                       % too dark to carry a readable hue
    blueFrac(lit) = blue(lit) ./ chanSum(lit);

    discColour = [];
    if hasAnatomy && discR > 0
        reach = discR * C.disc.peripapillaryMult * 1.6;
        near = discD2 <= reach^2;
        readable = near & inside & (blueFrac >= 0);

        if isempty(discTissue)
            whiteRef = readable & (discD2 <= discR^2*0.64);   % inner core
        else
            whiteRef = readable & discTissue;
        end
        % Retina reference: outside the rim and off the vessels, so neither the
        % disc's own edge nor a vessel's blood colour sets the other end.
        retinaRef = readable & (discD2 > (discR*1.6)^2) & ~vesselZone;
        if ~isempty(discTissue)
            retinaRef = retinaRef & ~discTissue;
        end

        if nnz(whiteRef) >= 200 && nnz(retinaRef) >= 200
            discF   = median(blueFrac(whiteRef));
            retinaF = median(blueFrac(retinaRef));
            if discF - retinaF >= C.disc.colourMinSeparation
                discColour.discF   = discF;
                discColour.retinaF = retinaF;
                discColour.cut = retinaF + (discF - retinaF)*C.disc.colourLevel;
            end
        end
    end

    % ---- Edge artifacts ---------------------------------------------------
    ap = dr.apertureGeometry(lum);
    edgeR2 = (ap.R * L.edgeZoneFrac)^2;
    edgeArtifactArea = round(n * L.edgeArtifactAreaFrac);

    % The junction test is only meaningful at the scale vessels cross and branch
    % at. A large blot hemorrhage genuinely does have several vessels running out
    % of the area around it, and testing one would throw away a real and serious
    % finding, so anything wider than a couple of vessel widths is exempt.
    seLen = max(3, round(min(w,h)*L.linearSEFrac));
    junctionMaxArea = round(pi * (seLen*2.0)^2);
    junctionArmLen = seLen * 1.5;

    rejected = struct("vessel",0, "junction",0, "tooLarge",0, "tooSmall",0, ...
                      "disc",0, "discTissue",0, "discColour",0, "weak",0, ...
                      "streak",0, "noSeed",0, "macula",0, "smooth",0, ...
                      "edge",0, "unbounded",0);

    candidates = {};
    darkCovered   = false(h, w);
    brightCovered = false(h, w);

    % ---- Pass 1: grown regions -------------------------------------------
    classify(dr.regionFeatures(darkLoose,   darkDev,   grad, darkSeed),   true);
    classify(dr.regionFeatures(brightLoose, brightDev, grad, brightSeed), false);

    % ---- Pass 2: safety net ----------------------------------------------
    % Something that was seeded must not vanish because the region it grew into
    % failed a size or shape test; the evidence for the lesion was the seed, not
    % the extent. Any seed not covered by an accepted region is reconsidered on
    % its own extent, so the worst case is that a lesion is reported slightly
    % smaller than it really is, never that it goes unreported. Seeds cannot
    % reintroduce vessels, because vessels never seed.
    darkLeft   = darkSeed   & ~darkCovered;
    brightLeft = brightSeed & ~brightCovered;
    if any(darkLeft(:))
        classify(dr.regionFeatures(darkLeft,   darkDev,   grad, []), true);
    end
    if any(brightLeft(:))
        classify(dr.regionFeatures(brightLeft, brightDev, grad, []), false);
    end

    % ---- Collect ----------------------------------------------------------
    if isempty(candidates)
        cands = struct("area",{},"cx",{},"cy",{},"type",{},"onVessel",{}, ...
                       "pixels",{},"maxContrast",{},"meanGradient",{}, ...
                       "aspect",{},"fillRatio",{});
    else
        cands = [candidates{:}];
        [~, ord] = sort([cands.area], "descend");
        cands = cands(ord);
    end

    counts = struct();
    for k = C.lesionOrder
        counts.(k) = 0;
    end
    lesionPixels = 0;
    onVesselCount = 0;
    for i = 1:numel(cands)
        counts.(cands(i).type) = counts.(cands(i).type) + 1;
        lesionPixels = lesionPixels + cands(i).area;
        onVesselCount = onVesselCount + cands(i).onVessel;
    end

    out = struct( ...
        "candidates", {cands}, "counts", {counts}, "rejected", {rejected}, ...
        "w", w, "h", h, "retinaArea", retinaArea, ...
        "lesionPixels", lesionPixels, "onVesselCount", onVesselCount, ...
        "noiseDark", noiseDark, "noiseBright", noiseBright, "scales", {scales}, ...
        "discExcluded", hasAnatomy, "maculaExcluded", hasMacula, ...
        "discExtentMeasured", hasAnatomy && isfield(anatomy.disc,"measuredExtent") ...
                              && anatomy.disc.measuredExtent, ...
        "discTissueMask", ~isempty(discTissue), ...
        "discColourUsed", ~isempty(discColour));

% =========================================================================
% Nested helpers. Nested rather than local so they can read the maps above
% without every one of them being threaded through an argument list.
% =========================================================================

    function k = satRung(area)
        % Which rung of the ladder speaks for a region of this size: the first
        % element at least as wide as the region is across. Below that the region
        % has not saturated yet and the ratio would only measure how much of it
        % is still being filled in.
        rr = sqrt(max(area,1)/pi);
        k = 0;
        for kk = 1:numel(seeds.satSE)-1
            if seeds.satSE(kk) >= rr
                k = kk;
                return;
            end
        end
    end

    function g = growthRatio(c, dark, k)
        % Measured over a disc centred on the region rather than over the
        % region's own pixels, and only where the response clears a core level.
        %
        % Two reasons. The faint margins a region is grown out to carry almost no
        % top-hat response at any element length, so including them drives both
        % means toward zero and the ratio toward noise. More importantly the
        % region's pixel list is not the structure: vessel pixels are removed
        % from what a region may grow into, so a lesion touching a vessel is
        % truncated, and what survives is the part furthest from the vessel - its
        % own rim, which is exactly the shape that reads as unbounded. Measured
        % that way the test rejected real lesions as soon as the vessel map
        % became dense enough to bite: on one retina it took five of them.
        if dark
            maps = seeds.satDark;   core = seeds.tDark*0.5;
        else
            maps = seeds.satBright; core = seeds.tBright*0.5;
        end
        small = maps{k};
        big   = maps{k+1};

        rad = max(3, round(sqrt(max(c.area,1)/pi)) + 1);
        x0 = max(1, round(c.cx)-rad);  x1 = min(w, round(c.cx)+rad);
        y0 = max(1, round(c.cy)-rad);  y1 = min(h, round(c.cy)+rad);

        sx = X(y0:y1, x0:x1);  sy = Y(y0:y1, x0:x1);
        disk = (sx - c.cx).^2 + (sy - c.cy).^2 <= rad^2;

        sIn  = inside(y0:y1, x0:x1);
        sVes = vesselMask(y0:y1, x0:x1);
        sSm  = small(y0:y1, x0:x1);
        sBg  = big(y0:y1, x0:x1);

        % Vessel pixels are excluded from the measurement, not merely from the
        % region. A lesion beside a vessel is otherwise measured partly on the
        % vessel, and a vessel is the one thing that keeps responding harder as
        % the element grows - so the lesion inherits its neighbour's growth and is
        % rejected for it. Measured on one retina the affected lesions came out
        % between 1.08 and 1.37 against a cut of 1.05, while lesions in open
        % retina sit at 1.00. This is the vessel map protecting a finding rather
        % than suppressing one.
        sel = disk & sIn & ~sVes & (sSm >= core);

        m = nnz(sel);
        a = sum(sSm(sel));
        if m < 3 || a <= 0
            g = [];                          % no responding core to judge
            return;
        end
        g = sum(sBg(sel)) / a;
    end

    function tf = failsSaturation(c, dark)
        % Bounded at any scale is bounded, and a region wider than the longest
        % element is exempt because there the test is silent rather than negative.
        if c.area > satMaxArea
            tf = false;  return;
        end
        k = satRung(c.area);
        if k < 1
            tf = false;  return;
        end
        g = growthRatio(c, dark, k);
        if isempty(g)
            tf = false;  return;
        end
        tf = g > L.saturationMaxGrowth;
    end

    function f = discTissueFrac(c)
        if isempty(discTissue)
            f = 0;
        else
            f = nnz(discTissue(c.pixels)) / numel(c.pixels);
        end
    end

    function tf = looksLikeDiscTissue(c)
        tf = false;
        if isempty(discColour)
            return;
        end
        v = blueFrac(c.pixels);
        v = v(v >= 0);
        if numel(v) < 3
            return;
        end
        tf = mean(v) >= discColour.cut;
    end

    function classify(comps, dark)
        for ci = 1:numel(comps)
            c = comps(ci);

            % No directional response anywhere in the region means no lesion.
            % This one test excludes the vessels, the macula and smooth
            % background shading together, without needing to know where any of
            % them are.
            if ~c.hasSeed,              rejected.noSeed   = rejected.noSeed+1;   continue; end
            if c.area < minArea,        rejected.tooSmall = rejected.tooSmall+1; continue; end
            if c.area > maxArea,        rejected.tooLarge = rejected.tooLarge+1; continue; end

            if hasAnatomy
                d2 = (c.cx - anatomy.disc.x)^2 + (c.cy - anatomy.disc.y)^2;
            else
                d2 = Inf;
            end
            if d2 <= (discR*1.15)^2
                rejected.disc = rejected.disc + 1;  continue;
            end

            % Outside that circle the disc can still be the explanation for a
            % bright mark: the rim it was grown from, a crescent of sclera beside
            % it, the nerve-fibre reflex arcing off it. Two independent ways of
            % saying so, either sufficient - the mark is built out of measured
            % disc tissue, or it is the colour of disc tissue rather than of
            % exudate.
            %
            % Dark candidates are deliberately exempt. Nothing about the disc is
            % dark, so a dark mark beside it is a hemorrhage and has to survive;
            % and the disc's margin is where a disc hemorrhage sits, which is a
            % finding worth more than everything this rule removes.
            if ~dark && d2 <= (discR*C.disc.peripapillaryMult)^2
                if discTissueFrac(c) >= C.disc.tissueOverlap
                    rejected.discTissue = rejected.discTissue + 1;  continue;
                end
                if looksLikeDiscTissue(c)
                    rejected.discColour = rejected.discColour + 1;  continue;
                end
            end

            % A large region far out toward the aperture rim is an artifact, not
            % a lesion: a shadow from pupil misalignment, or peripapillary
            % vignetting. Genuine peripheral lesions are small, so the rule is
            % keyed on size as well as on position and small ones stay.
            if c.area >= edgeArtifactArea
                de2 = (c.cx - ap.cx)^2 + (c.cy - ap.cy)^2;
                if de2 > edgeR2
                    rejected.edge = rejected.edge + 1;  continue;
                end
            end

            if c.aspect > 4.0 && c.fillRatio < 0.30
                rejected.streak = rejected.streak + 1;  continue;
            end

            % A region lying almost entirely on the vessel map is a piece of
            % vessel, whatever seeded it. The threshold is high on purpose: a
            % lesion touching a vessel overlaps it partially and must survive.
            if nnz(vesselZone(c.pixels))/numel(c.pixels) > L.vesselOverlapReject
                rejected.vessel = rejected.vessel + 1;  continue;
            end

            % Crossings, bifurcations, bends and choroidal mottling answer the
            % roundness test at one element length exactly as a lesion does. They
            % are told apart by whether the response stops growing when the
            % element does: a lesion is bounded, they are not.
            if failsSaturation(c, dark)
                rejected.unbounded = rejected.unbounded + 1;  continue;
            end

            onVessel = false;
            if dark && c.area <= junctionMaxArea
                ctx = dr.vesselArms(c.cx, c.cy, c.area, vesselZone, junctionArmLen);
                if dr.isVesselJunction(ctx)
                    rejected.junction = rejected.junction + 1;  continue;
                end
                onVessel = dr.isThroughVessel(ctx) || ctx.arms >= 1;
            end

            if dark
                floorV = L.darkContrastFloor;    noise = noiseDark;
            else
                floorV = L.brightContrastFloor;  noise = noiseBright;
            end
            % Contrast is measured against a background that excludes vessels, so
            % a vessel scores well on it simply for being a vessel. A candidate on
            % the vasculature therefore has to clear a higher bar than one in open
            % retina before the same figure means the same thing.
            bar = max(floorV, noise*L.minContrastK);
            if onVessel
                bar = bar * L.onVesselContrastFactor;
            end
            if c.maxContrast < bar
                rejected.weak = rejected.weak + 1;  continue;
            end

            if dark
                if c.area <= maMaxArea && c.aspect <= 2.1 && c.fillRatio >= 0.45
                    typeKey = "MA";
                else
                    typeKey = "HEM";
                end
            elseif c.meanGradient >= sharpEdgeThresh
                typeKey = "HE";
            elseif c.area >= cwsMinArea
                typeKey = "CWS";
            else
                rejected.weak = rejected.weak + 1;  continue;
            end

            rec = struct("area", c.area, "cx", c.cx, "cy", c.cy, ...
                         "type", typeKey, "onVessel", onVessel, ...
                         "pixels", {c.pixels}, "maxContrast", c.maxContrast, ...
                         "meanGradient", c.meanGradient, ...
                         "aspect", c.aspect, "fillRatio", c.fillRatio);
            candidates{end+1} = rec; %#ok<AGROW>

            if dark
                darkCovered(c.pixels) = true;
            else
                brightCovered(c.pixels) = true;
            end
        end
    end
end
