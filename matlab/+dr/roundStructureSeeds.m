function S = roundStructureSeeds(greenS, lumS, inside)
%ROUNDSTRUCTURESEEDS  Where lesions are, where vessels are, and at what scales.
%
%   S = dr.roundStructureSeeds(GREENS, LUMS, INSIDE) returns:
%     darkSeed, brightSeed   logical seeds for dark and bright lesions
%     vesselMask             logical vessel map
%     roundDark, roundBright the roundness responses the seeds were cut from
%     tDark, tBright         the thresholds used
%     satSE                  the ladder of element half-lengths
%     satDark, satBright     the responses at each rung, for the saturation test
%     seL                    the nominal element half-length
%
%   The seeds come from directional top-hats. Vessels vanish because they survive
%   a closing along their own direction; the macula vanishes because its gradual
%   ramp barely responds; a lesion sitting on a vessel is still found because the
%   lesion itself is round regardless of what it touches. That last property is
%   why this and not a plain top-hat: a square element removes every small lesion
%   along with the vessels, because a microaneurysm is smaller than a vessel is
%   wide and no size threshold can separate them.
%
%   The vessel map is derived from the same evidence rather than from a separate
%   pass, which is both cheaper and sounder. A vessel is elongated: the spread of
%   the closing across orientations outruns what every direction filled in common.
%   Comparing the two responses to each other rather than to a fixed number is
%   scale-free, so one rule covers a thin peripheral capillary and a wide vein at
%   the disc alike. The previous version thresholded each response against an
%   absolute level and recovered under a quarter of the vessel tree in 755
%   disconnected fragments; three quarters of the vasculature was simply not
%   being suppressed, which is where the marks on healthy retina came from.
%
%   ---- Why seeding stays on the shortest element ----
%   The obvious improvement was measured and rejected. A closing only fills a
%   structure narrower than the element, so a blot hemorrhage wider than the
%   element barely responds: on a planted hemorrhage of radius 20 the mean
%   top-hat at the nominal element is 0.4 grey levels and not one of its 1257
%   pixels seeds, while at twice and four times that length it answers at 25.4
%   and 34.7. Seeding from the longer rungs as well does fix the seeding, and
%   does not fix the detection - the newly seeded regions are rejected at the
%   contrast test instead, because a long element also responds to the macula and
%   to illumination falloff, which arrive together with the hemorrhages. Over
%   three retinas it raised false marks on healthy images from 19 to 29 while
%   recovering one large hemorrhage out of six. The gap is real, but it lives
%   further down in how extent and contrast are measured, and adding scales to
%   the seeder only moves the failure rather than removing it.

    C = dr.Config();
    L = C.lesion;
    [h, w] = size(greenS);

    seL = max(3, round(min(w,h) * L.linearSEFrac));

    [closedMin, closedMax] = dr.directionalMorph(greenS, seL, "close", L.linearOrientations);
    [~, openedMax]         = dr.directionalMorph(lumS,   seL, "open",  L.linearOrientations);

    % Filled by every direction: round. This is the lesion evidence.
    roundDark   = max(0, closedMin - greenS);
    roundBright = max(0, lumS - openedMax);
    % Spread across orientations: near zero on a round blob, large on a vessel.
    anisoDark   = max(0, closedMax - closedMin);

    roundDark(~inside)   = 0;
    roundBright(~inside) = 0;
    anisoDark(~inside)   = 0;

    % ---- The saturation ladder -----------------------------------------
    % The same two top-hats at successively longer elements. Everything bounded
    % has already been filled at its own scale and does not respond any harder at
    % the next rung; everything that continues past the element does. The
    % difference between consecutive rungs is the saturation test.
    satSE = seL;
    for k = 1:L.saturationLadder-1
        satSE(end+1) = max(satSE(end)+1, round(seL * L.saturationSEMult^k)); %#ok<AGROW>
    end

    satDark   = {roundDark};
    satBright = {roundBright};
    for k = 2:numel(satSE)
        [cMin, ~] = dr.directionalMorph(greenS, satSE(k), "close", L.saturationOrientations);
        [~, oMax] = dr.directionalMorph(lumS,   satSE(k), "open",  L.saturationOrientations);
        dk = max(0, cMin - greenS);   dk(~inside) = 0;
        br = max(0, lumS - oMax);     br(~inside) = 0;
        satDark{end+1}   = dk; %#ok<AGROW>
        satBright{end+1} = br; %#ok<AGROW>
    end

    % ---- Seeds -----------------------------------------------------------
    tDark   = max(L.seedFloorDark,   dr.tailQuantile(roundDark,   inside, L.seedTailQ));
    tBright = max(L.seedFloorBright, dr.tailQuantile(roundBright, inside, L.seedTailQ));

    darkSeed   = inside & (roundDark   > tDark);
    brightSeed = inside & (roundBright > tBright);

    % ---- Vessel map ------------------------------------------------------
    % Found at a strict level and followed outward at a permissive one, so a
    % vessel is traced along its length instead of appearing only where it
    % happens to be darkest. The elongation test comes first: a lesion fails it
    % however dark it is, which is what stops the vessel map from swallowing the
    % findings it exists to protect.
    tSeed = max(3, dr.tailQuantile(anisoDark, inside, L.vesselSeedQ));
    tGrow = max(2, dr.tailQuantile(anisoDark, inside, L.vesselGrowQ));

    elongated = inside & (anisoDark > L.vesselElongation * roundDark);
    vSeed = elongated & (anisoDark > tSeed);
    vGrow = elongated & (anisoDark > tGrow);
    vesselMask = dr.hysteresisMask(vSeed, vGrow);

    S = struct( ...
        "darkSeed", darkSeed, "brightSeed", brightSeed, ...
        "vesselMask", vesselMask, "seL", seL, ...
        "satSE", satSE, "satDark", {satDark}, "satBright", {satBright}, ...
        "roundDark", roundDark, "roundBright", roundBright, ...
        "tDark", tDark, "tBright", tBright);
end
