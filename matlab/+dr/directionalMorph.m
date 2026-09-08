function [accMin, accMax] = directionalMorph(A, radius, mode, orientCount)
%DIRECTIONALMORPH  Linear closing or opening at many orientations at once.
%
%   [MN, MX] = dr.directionalMorph(A, RADIUS, MODE, ORIENTCOUNT) applies a
%   morphological closing ("close") or opening ("open") along a line structuring
%   element of length 2*RADIUS+1, once per orientation, and returns the
%   elementwise minimum and maximum of the results across orientations.
%
%   This one operation is the whole basis on which lesions, vessels and
%   background are separated, so it is worth being precise about what the two
%   outputs mean.
%
%   A closing along a line fills any structure narrower than the element. At a
%   vessel, the element that happens to lie along it fills nothing, while every
%   element crossing it fills it completely. At a round lesion, every orientation
%   fills it about equally. So:
%
%     MN  the response every direction agreed on. Large only where the structure
%         is bounded in all directions — a lesion. This is the roundness map.
%     MX  the response at least one direction produced. Large wherever anything
%         thin exists, vessels included.
%
%   Their difference is the anisotropy, and comparing it to MN rather than to a
%   fixed number is what makes the vessel test scale-free: one rule covers a thin
%   peripheral capillary and a wide vein at the disc alike.
%
%   Orientations are spread over 180 degrees, since a line element is symmetric.
%   The sign of the angle is immaterial here — the family {0, 180/N, ...} is the
%   same set whether the angle is read clockwise or anticlockwise — so the image
%   coordinate convention does not need reconciling with STREL's.

    if nargin < 4 || isempty(orientCount)
        C = dr.Config();
        orientCount = C.lesion.linearOrientations;
    end
    A = double(A);
    radius = max(1, round(radius));
    len = 2*radius + 1;
    closing = (mode == "close");

    accMin = [];
    accMax = [];

    for o = 0:orientCount-1
        angleDeg = 180 * o / orientCount;
        se = strel("line", len, angleDeg);

        if closing
            r = imclose(A, se);      % dilate then erode: fills what is narrow
        else
            r = imopen(A, se);       % erode then dilate: removes what is narrow
        end

        if isempty(accMin)
            accMin = r;
            accMax = r;
        else
            accMin = min(accMin, r);
            accMax = max(accMax, r);
        end
    end

    if isempty(accMin)               % orientCount of zero: nothing was asked
        accMin = A;
        accMax = A;
    end
end
