function out = maskedBlur(A, mask, radius)
%MASKEDBLUR  Local mean over the masked pixels only.
%
%   OUT = dr.maskedBlur(A, MASK, RADIUS) averages A over a square window but
%   counts only pixels where MASK is true, renormalising each window by how many
%   of those it actually contained. Where a window holds no masked pixel at all
%   the original value is passed through unchanged.
%
%   This is the background estimator the whole detector rests on. A plain box
%   blur near the field-of-view rim averages retina together with the black
%   surround; the background comes out too low, ordinary edge retina reads as
%   brighter than its neighbourhood, and a ring of bright false findings appears
%   around the aperture. Dividing the blurred signal by the blurred mask
%   renormalises each window to the pixels that carry image data, which removes
%   the ring without eroding the field and losing genuine peripheral lesions.

    A = double(A);
    m = double(logical(mask));

    num = dr.boxBlur(A .* m, radius);
    den = dr.boxBlur(m, radius);

    out = A;                                % windows with no data keep the source
    valid = den > 1e-3;
    out(valid) = num(valid) ./ den(valid);
end
