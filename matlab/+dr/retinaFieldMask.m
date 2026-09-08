function mask = retinaFieldMask(lum, erodeFrac)
%RETINAFIELDMASK  The usable retina, found by fitting the aperture.
%
%   MASK = dr.retinaFieldMask(LUM, ERODEFRAC) returns a logical mask of the
%   retina, trimmed inward by ERODEFRAC of the fitted aperture radius.
%
%   Two shortcuts were tried before this and both failed on real images. A fixed
%   brightness cutoff keeps the vignetted rim, which is genuine retina but far
%   darker than the rest, so it reads as one enormous dark lesion — a crescent
%   appearing beside the disc. A fixed erosion cannot reach that rim either,
%   because the vignette is much wider than any sensible margin. Fitting the
%   aperture and working inside a fraction of its radius handles both, and adapts
%   to how much of the frame the retina fills.
%
%   Two thresholds are used, for two different jobs. Fitting the circle wants a
%   firm cutoff so the dim rim does not stretch it. Deciding membership wants a
%   far lower one, because a hemorrhage is dark: judged at the fitting cutoff it
%   falls below threshold and is carved out of the field as though it were
%   outside the camera's view, so every dark lesion punched a hole in its own
%   analysis region and could never be found.
%
%   The fraction kept is deliberately generous. Trimming a tenth of the radius
%   was measured to discard genuine peripheral lesions, so large rim artifacts
%   are dealt with by a size-keyed rule during classification instead.

    lum = double(lum);
    [h, w] = size(lum);
    hi = max(lum(:));

    fitThr    = max(14, hi * 0.20);
    memberThr = max(6,  hi * 0.06);

    lit = lum > fitThr;
    mask = false(h, w);
    if ~any(lit(:))
        return;
    end

    cols = find(any(lit, 1));
    rows = find(any(lit, 2));
    cx = (cols(1) + cols(end)) / 2;
    cy = (rows(1) + rows(end)) / 2;
    R  = max(cols(end) - cols(1), rows(end) - rows(1)) / 2;

    rIn = R * (1 - max(0.02, erodeFrac * 0.9));

    [X, Y] = meshgrid(1:w, 1:h);
    inCircle = (X - cx).^2 + (Y - cy).^2 <= rIn^2;

    mask = inCircle & (lum > memberThr);     % true black surround only
    mask = dr.erodeMask(mask, 2);
end
