function out = erodeMask(mask, radius)
%ERODEMASK  Erode a binary mask by a square structuring element.
%
%   OUT = dr.erodeMask(MASK, RADIUS) erodes by a (2*RADIUS+1) square.
%
%   The border is padded by replication rather than by IMERODE's default, which
%   assumes true outside the frame. Here the mask is a camera aperture that can
%   legitimately run off the edge of the image, and assuming retina outside the
%   frame leaves an un-eroded strip along any edge the aperture is clipped by.

    radius = round(radius);
    if radius <= 0
        out = logical(mask);
        return;
    end
    padded = padarray(logical(mask), [radius radius], "replicate", "both");
    eroded = imerode(padded, strel("square", 2*radius + 1));
    out = eroded(radius+1:end-radius, radius+1:end-radius);
end
