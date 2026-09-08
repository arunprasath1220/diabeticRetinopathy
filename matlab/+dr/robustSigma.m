function s = robustSigma(dev, mask)
%ROBUSTSIGMA  Noise scale of a deviation map, via the median absolute deviation.
%
%   S = dr.robustSigma(DEV, MASK) returns 1.4826*MAD of DEV over the masked
%   pixels, floored at half a grey level.
%
%   The standard deviation is the wrong statistic here and using it was a real
%   bug, not a stylistic choice. The vessel tree is a large population of strong
%   dark deviations; it inflates the standard deviation, which pushes every
%   threshold derived from it upward, which hides exactly the faint lesions the
%   detector exists to find. The median absolute deviation ignores that minority
%   and tracks the actual noise floor.
%
%   The 1.4826 makes MAD a consistent estimator of sigma for Gaussian data, so
%   the multipliers in DR.CONFIG can be read as "so many sigma" and mean it.

    v = double(dev(logical(mask)));
    if isempty(v)
        s = 1;
        return;
    end
    s = max(0.5, 1.4826 * mad(v, 1));    % mad(...,1) is the *median* AD
end
