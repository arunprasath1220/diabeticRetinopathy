function out = dilateMask(mask, radius)
%DILATEMASK  Dilate a binary mask by a square structuring element.
%
%   OUT = dr.dilateMask(MASK, RADIUS) dilates by a (2*RADIUS+1) square. Used to
%   turn the vessel map into a vessel *zone*: a lesion sitting against a vessel
%   must still be allowed to touch it, so the zone is only a pixel or two wider
%   than the map itself.

    radius = round(radius);
    if radius <= 0
        out = logical(mask);
        return;
    end
    padded = padarray(logical(mask), [radius radius], "replicate", "both");
    dilated = imdilate(padded, strel("square", 2*radius + 1));
    out = dilated(radius+1:end-radius, radius+1:end-radius);
end
