function out = boxBlur(A, radius)
%BOXBLUR  Mean over a (2*radius+1) square window, replicating at the border.
%
%   OUT = dr.boxBlur(A, RADIUS) is IMBOXFILT with the window expressed as a
%   radius rather than a side length, and with replicate padding fixed.
%
%   The padding is not a detail. The default is zero padding, and near the edge
%   of a fundus aperture that mixes the black surround into every window, drags
%   the local background down, and makes ordinary edge retina read as abnormally
%   bright — a ring of spurious findings around the border. Replicate padding
%   removes the artifact for a rectangular border; DR.MASKEDBLUR removes it for
%   the circular one, which is the boundary that actually matters here.

    if radius <= 0
        out = double(A);
        return;
    end
    out = imboxfilt(double(A), 2*radius + 1, "Padding", "replicate");
end
