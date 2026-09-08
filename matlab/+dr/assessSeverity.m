function sev = assessSeverity(lesions, anatomy, grading, quality)
%ASSESSSEVERITY  Step 5: apply the ICDR rule to what Step 4 found.
%
%   SEV = dr.assessSeverity(LESIONS, ANATOMY, GRADING, QUALITY)
%
%   The published scale is defined by which lesion types are present and, for
%   severe disease, by how many hemorrhages appear in each quadrant:
%
%     0  No apparent retinopathy   no abnormality
%     1  Mild NPDR                 microaneurysms only
%     2  Moderate NPDR             more than microaneurysms, less than severe
%     3  Severe NPDR               no PDR signs, and any of the 4-2-1 rule:
%                                    >20 intraretinal hemorrhages in each of 4
%                                    quadrants, venous beading in 2 or more, or
%                                    prominent IRMA in 1 or more
%     4  Proliferative DR          neovascularisation, or vitreous/preretinal
%                                    hemorrhage
%
%   Applying it here is legitimate because the rule is itself a function of
%   lesion type and count, which is exactly what Step 4 produces. What it is NOT
%   is a diagnosis: the counts are unvalidated candidates, so the grade inherits
%   every one of their errors.
%
%   ---- The hard ceiling ----
%   Two of the three arms of the 4-2-1 rule are venous beading and IRMA, and
%   level 4 is defined by neovascularisation. This detector cannot see any of the
%   three, because all are elongated structures that its vessel test removes by
%   construction. So it can reach level 3 only through the hemorrhage arm, can
%   never reach level 4, and - the part that matters clinically - can never rule
%   either of them out. That limitation is returned with every result rather than
%   buried, because a low grade here does not mean a low grade in the eye.

    C = dr.Config();

    if isempty(lesions)
        counts = [];
        ma = 0; hem = 0; he = 0; cws = 0;
    else
        counts = lesions.counts;
        ma  = counts.MA;   hem = counts.HEM;
        he  = counts.HE;   cws = counts.CWS;
    end
    beyondMA = hem + he + cws;
    total = ma + beyondMA;

    grid = dr.quadrantLesionCounts(lesions, anatomy);
    quadsOverThreshold = 0;
    if ~isempty(grid)
        for q = C.icdr.quadrantNames
            f = matlab.lang.makeValidName(q);
            if grid.HEM.(f) > C.icdr.hemPerQuadrant
                quadsOverThreshold = quadsOverThreshold + 1;
            end
        end
    end
    hemArmMet = ~isempty(grid) && quadsOverThreshold == 4;

    basis = string.empty(1,0);
    if isempty(lesions)
        level = NaN;  label = "Not assessed";
        basis(end+1) = "lesion detection did not run on this image";
    elseif hemArmMet
        level = 3;  label = "Severe NPDR pattern";
        basis(end+1) = sprintf(['more than %d hemorrhage candidates in each of ' ...
            'the four quadrants, which is the hemorrhage arm of the 4-2-1 rule'], ...
            C.icdr.hemPerQuadrant);
    elseif beyondMA > 0
        level = 2;  label = "Moderate NPDR pattern";
        basis(end+1) = sprintf(['more than microaneurysms alone: %d hemorrhage, ' ...
            '%d hard exudate and %d cotton wool candidates'], hem, he, cws);
    elseif ma > 0
        level = 1;  label = "Mild NPDR pattern";
        basis(end+1) = sprintf("%d microaneurysm candidate(s) and nothing else", ma);
    else
        level = 0;  label = "No retinopathy observed";
        basis(end+1) = "no lesion candidate passed the detector on this image";
    end

    % ---- Everything that makes this estimate untrustworthy here ----------
    doubts = string.empty(1,0);
    if isempty(lesions)
        doubts(end+1) = "lesion detection did not complete";
    end
    if isempty(anatomy)
        doubts(end+1) = "landmarks could not be estimated, so the quadrant rule " + ...
                        "for severe disease could not be applied at all";
    end
    if ~isempty(quality) && isfield(quality,"verdict") && quality.verdict == "enhance"
        doubts(end+1) = "image quality was borderline and had to be enhanced " + ...
                        "before analysis";
    end
    if ~isempty(lesions) && total > C.lesion.noiseSuspicionCount
        doubts(end+1) = "the candidate count is high enough to suggest the " + ...
                        "detector is responding to image noise";
    end
    if ~isempty(grid) && quadsOverThreshold > 0 && quadsOverThreshold < 4
        doubts(end+1) = sprintf(['hemorrhage candidates exceed the severe-disease ' ...
            'threshold in %d of four quadrants, which sits right on the boundary ' ...
            'of the rule'], quadsOverThreshold);
    end

    % ---- The cross-check the two independent stages exist to provide -----
    if ~isempty(grading)
        modelSaysDR = grading.predClass == 2;     % 1-based: 2 is "DR present"
        if modelSaysDR && level == 0
            doubts(end+1) = "the classifier reports disease present while the " + ...
                "detector found no lesion at all, and the two disagree";
        end
        if ~modelSaysDR && level >= 2
            doubts(end+1) = "the classifier reports no disease while the detector " + ...
                "found lesions beyond microaneurysms, and the two disagree";
        end
    else
        doubts(end+1) = "the classifier did not run, so there is no independent " + ...
                        "check on this result";
    end

    % Refer whenever the rule says referable, whenever anything is in doubt, and
    % whenever any lesion at all was seen. Only a clean, agreeing, lesion-free
    % image avoids it. In a screening programme the cost of the two errors is not
    % symmetric, and this is where that asymmetry is written down.
    referable = ~isnan(level) && level >= 2;
    uncertain = ~isempty(doubts);
    refer = referable || uncertain || (~isnan(level) && level >= 1);

    if referable
        reason = "This image reaches the referable threshold (moderate NPDR or " + ...
                 "worse) under the ICDR rule.";
    elseif uncertain
        reason = "This result is not reliable enough to stand on its own.";
    elseif ~isnan(level) && level >= 1
        reason = "Lesions were seen. Any retinopathy needs a specialist opinion.";
    else
        reason = "Severe disease and proliferative disease cannot be excluded " + ...
                 "by this method.";
    end

    sev = struct("level", level, "label", label, "basis", {basis}, ...
                 "doubts", {doubts}, "refer", refer, "referable", referable, ...
                 "uncertain", uncertain, "reason", reason, ...
                 "grid", {grid}, "quadsOverThreshold", quadsOverThreshold, ...
                 "counts", {counts}, "total", total, ...
                 "ceiling", "cannot detect venous beading, IRMA or " + ...
                            "neovascularisation, so level 4 is unreachable and " + ...
                            "neither it nor severe disease can be ruled out");
end
