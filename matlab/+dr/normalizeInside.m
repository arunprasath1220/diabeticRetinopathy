function out = normalizeInside(A, mask)
%NORMALIZEINSIDE  Scale to 0..1 using only the masked pixels' range.
%
%   OUT = dr.normalizeInside(A, MASK) maps A onto 0..1 using the minimum and
%   maximum found *inside* MASK, and returns zero everywhere outside it.
%
%   Normalising over the whole frame instead would let the black surround set the
%   bottom of the range, which compresses everything the retina actually contains
%   into the top of the scale and flattens the very differences being measured.

    A = double(A);
    mask = logical(mask);
    v = A(mask);
    if isempty(v)
        out = zeros(size(A));
        return;
    end
    lo = min(v);
    hi = max(v);
    range = hi - lo;
    if range == 0
        range = 1;
    end
    out = zeros(size(A));
    out(mask) = (A(mask) - lo) / range;
end
