function t = tailQuantile(map, mask, q)
%TAILQUANTILE  Upper-quantile anchor for a one-sided response map.
%
%   T = dr.tailQuantile(MAP, MASK, Q) is PRCTILE of MAP over the masked pixels
%   at the Q quantile.
%
%   DR.ROBUSTSIGMA is the right tool for the symmetric deviation maps, where the
%   median really is the background and the MAD really is the noise. It is the
%   wrong tool for a top-hat response, which is non-negative and piles most of
%   its mass on exactly zero: there the MAD describes that zero atom and says
%   nothing whatever about the tail the threshold has to sit in. This reads the
%   tail directly.

    v = double(map(logical(mask)));
    if isempty(v)
        t = 1;
        return;
    end
    t = prctile(v, q * 100);
end
