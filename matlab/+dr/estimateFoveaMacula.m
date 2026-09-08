function fm = estimateFoveaMacula(I, disc)
%ESTIMATEFOVEAMACULA  Step 3: locate the fovea, and with it the nasal side.
%
%   FM = dr.estimateFoveaMacula(I, DISC) returns fovea (x,y), maculaRadius,
%   nasalSide and a short statement of the evidence used.
%
%   Pure geometry from the disc puts the marker in roughly the right area but
%   rarely on the fovea itself, so the position is measured: the fovea is the
%   darkest part of the central retina, being avascular and pigment-dense. The
%   vessels are removed first by a grayscale closing at vessel scale, because
%   otherwise the darkest thing in the macula is a vein.
%
%   The search is anchored as well as measured. Darkness alone let any dark
%   structure in the window capture the marker; weighting it by distance from the
%   anatomically expected point - about two and a half disc diameters temporal,
%   a little below the disc's level - keeps the estimate where the fovea has to
%   be while still letting real macular darkening move it. The 0.4 floor on the
%   darkness term is what makes a featureless macula degrade to the anatomical
%   position instead of snapping to noise.

    % The macula is a broad feature, so the search runs on a downscaled copy.
    % That makes the morphology cheap and stops single dark pixels mattering.
    scale = min(1, 256 / max(size(I,1), size(I,2)));
    small = imresize(I, scale, "bilinear");
    ch = dr.imageChannels(small);
    w = ch.w;  h = ch.h;  lum = ch.lum;

    inside = dr.retinaFieldMask(lum, 0.06);

    dx = disc.x * scale;
    dy = disc.y * scale;
    dd = max(6, disc.radius * 2 * scale);

    % Which way is temporal: away from the frame centre, on the disc's side.
    if (size(I,2)/2 - disc.x) >= 0
        sgn = 1;
    else
        sgn = -1;
    end

    % ---- Remove the vessels outright ------------------------------------
    % A closing by a square element at vessel width. A square is right here and
    % wrong for lesion work: it removes everything smaller than itself, which is
    % exactly what is wanted when the target is a feature many times larger.
    vr = max(2, round(dd*0.16));
    vesselFree = imclose(lum, strel("square", 2*vr + 1));

    % ---- How much darker than the surrounding retina, at macular scale ---
    bg = dr.maskedBlur(vesselFree, inside, max(6, round(dd*1.5)));
    darkness = max(0, bg - vesselFree);
    darkness(~inside) = 0;
    dMax = max(darkness(:));
    if dMax <= 0
        dMax = 1;
    end

    % ---- Anatomical anchor ----------------------------------------------
    ex = dx + sgn*dd*2.5;
    ey = dy + dd*0.3;
    sigma = dd*0.8;
    twoSig2 = 2*sigma^2;
    reach = round(sigma*2);

    [X, Y] = meshgrid(1:w, 1:h);
    prior = exp(-((X-ex).^2 + (Y-ey).^2) / twoSig2);
    score = (0.4 + 0.6*(darkness/dMax)) .* prior;

    window = false(h, w);
    ys = max(1, round(ey-reach)) : min(h, round(ey+reach));
    xs = max(1, round(ex-reach)) : min(w, round(ex+reach));
    window(ys, xs) = true;
    window = window & inside;

    if any(window(:))
        scoreW = score;
        scoreW(~window) = -Inf;
        [best, bestIdx] = max(scoreW(:));
        [by, bx] = ind2sub([h w], bestIdx);

        % Score-weighted centroid, so the marker sits in the middle of the dark
        % area rather than on its single darkest pixel.
        box = round(dd*0.5);
        near = false(h, w);
        near(max(1,by-box):min(h,by+box), max(1,bx-box):min(w,bx+box)) = true;
        sel = near & inside & (score >= best*0.75);

        if any(sel(:))
            wt = score(sel);
            fx = sum(X(sel).*wt) / sum(wt);
            fy = sum(Y(sel).*wt) / sum(wt);
        else
            fx = bx;  fy = by;
        end
        fovea = [fx/scale, fy/scale];
        evidence = "measured - darkest macular region after vessel removal, " + ...
                   "anchored to the expected position relative to the disc";
    else
        fovea = [ex/scale, ey/scale];
        evidence = "geometry only - no usable retinal area in the expected " + ...
                   "zone, so the anatomical position is used unrefined";
    end

    margin = disc.radius * 1.2;
    fovea(1) = min(size(I,2)-margin, max(margin, fovea(1)));
    fovea(2) = min(size(I,1)-margin, max(margin, fovea(2)));

    % The disc is nasal to the fovea in either eye, so the side the disc sits on
    % is the nasal side - no need to know OD from OS.
    if (fovea(1) - disc.x) >= 0
        nasalSide = "left";
    else
        nasalSide = "right";
    end

    fm = struct("fovea", struct("x", fovea(1), "y", fovea(2)), ...
                "maculaRadius", disc.radius*2*1.1, ...
                "nasalSide", nasalSide, ...
                "evidence", evidence);
end
