function ap = apertureGeometry(lum)
%APERTUREGEOMETRY  Centre and radius of the camera aperture.
%
%   AP = dr.apertureGeometry(LUM) fits the lit circle from the extent of the
%   non-black area and returns AP.cx, AP.cy (1-based pixel coordinates) and AP.R.
%
%   The cutoff is deliberately firm — a fifth of the frame's peak — so that the
%   dim vignetted rim does not stretch the fitted circle outward. Deciding which
%   pixels are *inside* the aperture is a different question with a different
%   threshold; see DR.RETINAFIELDMASK.

    lum = double(lum);
    hi = max(lum(:));
    thr = max(14, hi * 0.20);

    lit = lum > thr;
    if ~any(lit(:))
        [h, w] = size(lum);
        ap = struct("cx", w/2, "cy", h/2, "R", min(w,h)/2);
        return;
    end

    cols = find(any(lit, 1));
    rows = find(any(lit, 2));
    minX = cols(1);   maxX = cols(end);
    minY = rows(1);   maxY = rows(end);

    ap.cx = (minX + maxX) / 2;
    ap.cy = (minY + maxY) / 2;
    ap.R  = max(1, max(maxX - minX, maxY - minY) / 2);
end
