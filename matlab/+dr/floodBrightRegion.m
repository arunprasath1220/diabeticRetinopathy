function R = floodBrightRegion(blur, inside, sx, sy, cut, maxR, maxArea)
%FLOODBRIGHTREGION  Grow the bright region containing a seed, under bounds.
%
%   R = dr.floodBrightRegion(BLUR, INSIDE, SX, SY, CUT, MAXR, MAXAREA) returns a
%   struct with fields mask, area, cx, cy for the 4-connected region of
%   BLUR >= CUT containing the seed (SX, SY), or [] if either bound is passed.
%
%   Returning nothing is a real answer here, not a failure path. Once the fill
%   has run past MAXR or MAXAREA it has left the structure it was asked to
%   measure and is spreading into open retina, and no number it could return
%   would mean anything. The caller keeps whatever estimate it already had, which
%   is worse but honest.
%
%   The radius bound is applied to the candidate set before reconstruction rather
%   than checked afterwards, which is what makes it a genuine barrier: a region
%   cannot reach around through distant pixels and come back.

    R = [];
    [h, w] = size(blur);
    sx = round(sx);  sy = round(sy);
    if sx < 1 || sy < 1 || sx > w || sy > h
        return;
    end
    if ~inside(sy, sx) || blur(sy, sx) < cut
        return;
    end

    [X, Y] = meshgrid(1:w, 1:h);
    withinReach = (X - sx).^2 + (Y - sy).^2 <= maxR^2;

    bw = logical(inside) & (blur >= cut) & withinReach;

    seed = false(h, w);
    seed(sy, sx) = true;

    mask = imreconstruct(seed, bw, 4);

    area = nnz(mask);
    if area == 0 || area > maxArea
        return;                              % escaped: refuse to answer
    end

    R.mask = mask;
    R.area = area;
    R.cx = mean(X(mask));
    R.cy = mean(Y(mask));
end
