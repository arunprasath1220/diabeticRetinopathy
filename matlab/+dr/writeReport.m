function txt = writeReport(R, outFile)
%WRITEREPORT  The clinical report, with its own limitations attached.
%
%   TXT = dr.writeReport(R) formats the result of DR.RUNPIPELINE as text.
%   TXT = dr.writeReport(R, FILE) also writes it to FILE.
%
%   Every section that could not be computed says so explicitly. That is the
%   whole design of this function: a report with a missing section is safe,
%   because the reader knows to go and look; a report that silently omits the
%   section it could not compute reads exactly like a normal result.

    C = dr.Config();
    L = string.empty(1,0);

    L(end+1) = "DIABETIC RETINOPATHY SCREENING - CANDIDATE REPORT";
    L(end+1) = "Generated " + string(R.timestamp);
    L(end+1) = string(repmat('=', 1, 68));
    L(end+1) = "";
    L(end+1) = "NOT A DIAGNOSIS. Every count below is an unvalidated candidate " + ...
               "produced by an automated detector, for review by an " + ...
               "ophthalmologist who makes the referral decision.";
    L(end+1) = "";

    % ---- Step 1 ---------------------------------------------------------
    L(end+1) = "STEP 1  IMAGE QUALITY";
    L(end+1) = sprintf("  focus %.1f | brightness %.1f +/- %.1f | field %.0f%%", ...
        R.metrics.focusVar, R.metrics.brightMean, R.metrics.brightStd, R.metrics.fov*100);
    L(end+1) = "  verdict: " + R.quality.verdict;
    if strlength(R.quality.reason) > 0
        L(end+1) = "  " + R.quality.reason;
    end
    L(end+1) = "";

    if R.stopped
        L(end+1) = "ANALYSIS STOPPED. " + R.severity.reason;
        L(end+1) = "Recapture if you can, and refer if a usable image cannot be obtained.";
        txt = strjoin(L, newline);
        writeIfAsked(txt, nargin, outFile);
        return;
    end

    % ---- Step 2 ---------------------------------------------------------
    L(end+1) = "STEP 2  CLASSIFIER";
    if isempty(R.grading)
        L(end+1) = "  NOT RUN - no classifier was available in this session.";
        L(end+1) = "  No grade and no attention map are shown, rather than " + ...
                   "placeholder numbers that would misrepresent the model.";
    else
        L(end+1) = sprintf("  P(no DR) = %.3f, P(DR present) = %.3f", ...
            R.grading.probs(1), R.grading.probs(2));
        L(end+1) = "  predicted: " + C.model.classNames(R.grading.predClass);
    end
    L(end+1) = "";

    % ---- Step 3 ---------------------------------------------------------
    L(end+1) = "STEP 3  ANATOMY AND EXPLANATION";
    if isempty(R.anatomy)
        L(end+1) = "  Landmarks could not be estimated on this image.";
    else
        a = R.anatomy;
        L(end+1) = sprintf("  optic disc  x %.0f y %.0f, radius %.0f px", ...
            a.disc.x, a.disc.y, a.disc.radius);
        if a.disc.measuredExtent
            L(end+1) = "    extent grown from the disc's own brightness down to " + ...
                       "the half-way level between it and the retina around it";
        else
            L(end+1) = "    extent could not be grown; radius is from the bright " + ...
                       "plateau and may undersize the disc";
        end
        L(end+1) = sprintf("  fovea       x %.0f y %.0f", a.fovea.x, a.fovea.y);
        L(end+1) = "    " + a.evidence;
    end
    if isfield(R, "camDescription")
        d = R.camDescription;
        L(end+1) = sprintf("  attention peak in the %s quadrant, concentration %.2f", ...
            d.region, d.concentration);
        if d.flat
            L(end+1) = "    the map is warm across the whole retina, which " + ...
                       "localises nothing - no regions are marked";
        elseif ~d.focal
            L(end+1) = "    attention is spread rather than concentrated";
        end
    end
    L(end+1) = "";

    % ---- Step 4 ---------------------------------------------------------
    L(end+1) = "STEP 4  LESION CANDIDATES";
    if isempty(R.lesions)
        L(end+1) = "  DETECTION FAILED on this image. No mask is shown, rather " + ...
                   "than an empty one being passed off as a clear retina.";
    else
        les = R.lesions;
        n = numel(les.candidates);
        L(end+1) = sprintf("  %d candidate regions, %.2f%% of retinal area", ...
            n, 100*les.lesionPixels/max(1,les.retinaArea));
        for k = C.lesionOrder
            L(end+1) = sprintf("    %-32s %d", C.lesionTypes.(k).label, les.counts.(k));
        end
        if les.onVesselCount > 0
            L(end+1) = sprintf(['  %d lie along a vessel and are struck through ' ...
                'on the overlay - treat those with extra caution, because a wide ' ...
                'spot in a vessel looks the same.'], les.onVesselCount);
        end

        rj = les.rejected;
        L(end+1) = "  rejected during filtering:";
        L(end+1) = sprintf("    %d vessel, %d junctions/bends, %d unbounded, %d faint", ...
            rj.vessel, rj.junction, rj.unbounded, rj.weak);
        L(end+1) = sprintf("    %d in the disc, %d built of disc tissue, %d disc-coloured", ...
            rj.disc, rj.discTissue, rj.discColour);
        L(end+1) = sprintf("    %d too large, %d too small, %d streaks, %d rim artifacts", ...
            rj.tooLarge, rj.tooSmall, rj.streak, rj.edge);

        if ~les.discExcluded
            L(end+1) = "  NOTE: no disc position, so the disc was not excluded and " + ...
                       "its bright pixels may appear as exudate candidates.";
        elseif ~les.discExtentMeasured
            L(end+1) = "  NOTE: the disc's extent could not be grown, so only the " + ...
                       "estimated circle excluded it - bright marks at its margin " + ...
                       "may have survived.";
        elseif ~les.discColourUsed
            L(end+1) = "  NOTE: disc and retina did not separate in colour, so " + ...
                       "peripapillary marks were judged on the measured extent alone.";
        end
        if n > C.lesion.noiseSuspicionCount
            L(end+1) = "  WARNING: this count is too high to be lesions. It almost " + ...
                       "always means the detector is firing on image noise or " + ...
                       "compression artefacts. Do not read it as a lesion burden.";
        end
    end
    L(end+1) = "";

    % ---- Step 5 ---------------------------------------------------------
    sev = R.severity;
    L(end+1) = "STEP 5  ICDR SEVERITY ESTIMATE";
    if isnan(sev.level)
        L(end+1) = "  " + sev.label;
    else
        L(end+1) = sprintf("  Level %d - %s", sev.level, sev.label);
    end
    for b = sev.basis
        L(end+1) = "    basis: " + b;
    end
    if ~isempty(sev.grid)
        L(end+1) = "  hemorrhage candidates per quadrant:";
        for q = C.icdr.quadrantNames
            f = matlab.lang.makeValidName(q);
            L(end+1) = sprintf("    %-22s %d", q, sev.grid.HEM.(f));
        end
    end
    L(end+1) = "";
    L(end+1) = "  CEILING: this detector " + sev.ceiling + ".";
    L(end+1) = "";

    if ~isempty(sev.doubts)
        L(end+1) = "  reasons for caution on this image:";
        for d = sev.doubts
            L(end+1) = "    - " + d;
        end
        L(end+1) = "";
    end

    % ---- Recommendation --------------------------------------------------
    L(end+1) = string(repmat('-', 1, 68));
    if sev.refer
        L(end+1) = "RECOMMENDATION: REFER for ophthalmologist review.";
    else
        L(end+1) = "RECOMMENDATION: no referral indicated by this analysis.";
    end
    L(end+1) = "  " + sev.reason;
    L(end+1) = string(repmat('-', 1, 68));

    txt = strjoin(L, newline);
    writeIfAsked(txt, nargin, outFile);
end

function writeIfAsked(txt, nIn, outFile)
    if nIn >= 2 && ~isempty(outFile)
        fid = fopen(outFile, "w");
        if fid < 0
            error("dr:writeReport:cannotWrite", "could not open %s", outFile);
        end
        cleanup = onCleanup(@() fclose(fid));
        fprintf(fid, "%s\n", txt);
    end
end
