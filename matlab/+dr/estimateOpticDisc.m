function disc = estimateOpticDisc(I)
%ESTIMATEOPTICDISC  Step 3: locate the optic disc and measure its extent.
%
%   DISC = dr.estimateOpticDisc(I) returns a struct with fields:
%     x, y            centre, in 1-based pixel coordinates
%     radius          disc radius in pixels
%     tissue          logical mask of the disc and its bright halo, or []
%     measuredExtent  true if the extent was grown rather than inferred
%     score           the brightness/texture score at the seed
%
%   Finding it. Brightness alone picks the wrong region whenever the photo
%   carries a specular highlight or a blown-out patch, so brightness is combined
%   with local contrast: the disc carries the vessel trunk and a sharp rim and is
%   therefore bright AND textured, while flare is bright and smooth.
%
%   Measuring it. This is where a real bug lived. The radius used to come from
%   the area of the plateau surviving a cut at a fixed height above the *global*
%   retinal mean, and both halves of that are wrong. The reference is global, so
%   a change in average brightness anywhere else in the frame moves the disc's
%   boundary; and the height is fixed, so on a soft-edged structure the contour
%   sits inside the rim rather than on it. Both push the same way, and the
%   measured radius came out a few per cent small.
%
%   A few per cent sounds harmless until it is multiplied by the 1.15 exclusion
%   factor and asked to reach past the rim, which it does not. The rim itself, a
%   scleral crescent beside it and the peripapillary reflex all fall in the gap
%   between where the circle stops and where the disc actually ends - and every
%   one of them is bright, round and sharply bounded, which is the description of
%   a hard exudate. They were duly reported as one.
%
%   So the extent is grown instead, from a level referred to the retina
%   immediately around the disc, and placed at the half-maximum point where the
%   contour of a blurred edge actually sits. Twice: once for the disc proper, and
%   once lower for the halo it shades into.
%
%   See also DR.FLOODBRIGHTREGION, DR.DETECTLESIONCANDIDATES.

    C = dr.Config();
    ch = dr.imageChannels(I);
    w = ch.w;  h = ch.h;
    lum = ch.lum;

    inside = dr.retinaFieldMask(lum, 0.03);
    if ~any(inside(:))
        error("dr:estimateOpticDisc:noField", ...
              "no usable retinal area for disc estimation");
    end

    % ---- Seed: bright AND textured -------------------------------------
    r = max(5, round(min(w,h) * 0.045));
    blur = dr.maskedBlur(lum, inside, r);

    blurSq = dr.maskedBlur(lum.^2, inside, r);
    sd = sqrt(max(0, blurSq - blur.^2));

    nb = dr.normalizeInside(blur, inside);
    ns = dr.normalizeInside(sd, inside);

    score = nb*0.65 + ns*0.35;
    score(~inside) = -Inf;
    [best, bestIdx] = max(score(:));
    [seedY, seedX] = ind2sub([h w], bestIdx);

    % ---- The old plateau estimate, kept as the fallback -----------------
    meanBlur = mean(blur(inside));
    peak = blur(bestIdx);
    cut = peak - (peak - meanBlur) * 0.35;

    box = round(min(w,h) * 0.16);
    ys = max(1, seedY-box) : min(h, seedY+box);
    xs = max(1, seedX-box) : min(w, seedX+box);
    sub = false(h, w);
    sub(ys, xs) = true;
    plateau = sub & inside & (blur >= cut);

    rMin = min(w,h) * 0.035;
    rMax = min(w,h) * 0.11;
    [X, Y] = meshgrid(1:w, 1:h);

    area = nnz(plateau);
    if area > 0
        discX = mean(X(plateau));
        discY = mean(Y(plateau));
    else
        discX = seedX;
        discY = seedY;
    end
    radius = min(rMax, max(rMin, sqrt(max(area,1)/pi)));

    % ---- Measured extent ------------------------------------------------
    cx0 = min(w, max(1, round(discX)));
    cy0 = min(h, max(1, round(discY)));
    r0 = radius;

    % Peripapillary background: retina in an annulus outside the disc, close
    % enough to share its illumination. The inner edge is held clear of the blur
    % radius as well as of the disc, because the same blur that makes the disc a
    % plateau also spreads it outward, and an annulus inside that spill would
    % read the disc back as its own background and shrink the fill to nothing.
    ringInner = max(r0*2.5, r0 + r*1.5);
    ringOuter = max(ringInner*1.6, r0*4.0);
    d2 = (X - cx0).^2 + (Y - cy0).^2;
    ringMask = inside & (d2 >= ringInner^2) & (d2 <= ringOuter^2);

    if nnz(ringMask) > 200
        ppBg = median(blur(ringMask));
    else
        % The disc sits near the edge of the aperture and no annulus fits. The
        % global mean is a worse reference, but it errs toward a smaller fill
        % rather than a runaway one, which is the safe direction.
        ppBg = meanBlur;
    end

    % The disc's own level, as a high quantile of its core rather than the single
    % brightest pixel in it, so one specular speck on the cup cannot set the
    % scale that both cuts are measured against.
    coreMask = inside & (d2 <= max(2, r0*0.6)^2);
    if any(coreMask(:))
        discLevel = prctile(blur(coreMask), 75);
    else
        discLevel = peak;
    end

    tissue = [];
    measuredExtent = false;
    span = discLevel - ppBg;

    % A span of a grey level or less means the disc is not separable from the
    % retina around it on this frame. Anything grown from that is noise, so
    % nothing is grown and the plateau estimate is left to stand alone.
    if span > 1
        maxR = r0 * C.disc.fillMaxMult;
        maxArea = pi * (r0*2.8)^2;

        edge = dr.floodBrightRegion(blur, inside, cx0, cy0, ...
                                    ppBg + span*C.disc.edgeLevel, maxR, maxArea);
        if ~isempty(edge)
            radius = min(rMax, max(rMin, sqrt(edge.area/pi)));
            discX = edge.cx;
            discY = edge.cy;
            tissue = edge.mask;
            measuredExtent = true;
        end

        % The halo takes in whatever the disc shades into: the rim it was grown
        % from, a scleral crescent beside it, the nerve-fibre reflex arcing off
        % it. It supersedes the edge mask when it can be grown, being the larger
        % of the two and the one the bright-candidate test actually wants.
        halo = dr.floodBrightRegion(blur, inside, cx0, cy0, ...
                                    ppBg + span*C.disc.haloLevel, maxR, maxArea);
        if ~isempty(halo)
            tissue = halo.mask;
        end
    end

    disc = struct( ...
        "x", discX, "y", discY, "radius", radius, "score", best, ...
        "tissue", {tissue}, "measuredExtent", measuredExtent);
end
