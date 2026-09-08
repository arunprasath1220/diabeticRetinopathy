function q = assessQuality(m)
%ASSESSQUALITY  Step 1: the gradable / ungradable decision.
%
%   Q = dr.assessQuality(M) turns the metrics from DR.COMPUTEMETRICS into one of
%   three verdicts, with the reason attached:
%
%     "reject"   nothing about this image can be assessed, and the pipeline stops
%     "enhance"  borderline, so run the enhancement stage before analysing
%     "ok"       analyse as captured
%
%   The rejection path matters more than it looks. Screening an unassessable
%   image produces a confident-looking result with nothing underneath it, and in
%   a rural screening programme that result is the one that gets acted on. The
%   reasons are phrased as instructions to whoever is holding the camera, because
%   that is the only person who can fix the problem.

    C = dr.Config();
    T = C.quality;

    if m.fov < T.minFov
        q = reject(sprintf(['Insufficient retinal field of view (only %.0f%% of ' ...
            'the frame is non-black) - recentre the eye in the camera and recapture.'], ...
            m.fov*100));
        return;
    end
    if m.brightMean < T.minBrightMean
        q = reject(sprintf(['Mean brightness too low (%.1f/255) - recapture with ' ...
            'more illumination.'], m.brightMean));
        return;
    end
    if m.brightMean > T.maxBrightMean
        q = reject(sprintf(['Image overexposed (mean brightness %.1f/255) - reduce ' ...
            'flash intensity and recapture.'], m.brightMean));
        return;
    end
    if m.focusVar < T.minFocusVar
        q = reject(sprintf(['Image too blurry (focus score %.1f) - hold the camera ' ...
            'steady and refocus before recapture.'], m.focusVar));
        return;
    end

    reasons = string.empty(1,0);
    if m.focusVar < T.enhanceFocusVar
        reasons(end+1) = "borderline focus";
    end
    if m.brightMean < T.enhanceBrightLow || m.brightMean > T.enhanceBrightHigh
        reasons(end+1) = "borderline exposure";
    end
    if m.fov < T.enhanceFov
        reasons(end+1) = "limited field of view";
    end

    if ~isempty(reasons)
        q = struct("verdict", "enhance", "reason", strjoin(reasons, ", "), ...
                   "reasons", reasons);
    else
        q = struct("verdict", "ok", "reason", "", "reasons", string.empty(1,0));
    end
end

function q = reject(reason)
    q = struct("verdict", "reject", "reason", string(reason), ...
               "reasons", string.empty(1,0));
end
