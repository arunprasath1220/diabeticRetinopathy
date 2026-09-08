function d = describeGradCAM(cam, anatomy)
%DESCRIBEGRADCAM  Say in words what the attention map looks like.
%
%   D = dr.describeGradCAM(CAM, ANATOMY) returns region, focal, flat,
%   concentration, median, peakX, peakY.
%
%   CONCENTRATION is the share of total activation held by the warmest tenth of
%   the retina. It is what separates a map with one hotspot from a map that is
%   uniformly warm - and the peak value cannot make that distinction, because
%   normalisation guarantees a peak of 1.0 either way.
%
%   Every statistic is over retina only; see DR.CONFINECAMTORETINA for why.

    camArray = cam.camArray;
    inside = cam.inside;

    v = camArray(inside);
    if isempty(v)
        d = struct("region","n/a", "focal",false, "flat",true, ...
                   "concentration",0, "median",0, "peakX",1, "peakY",1);
        return;
    end

    masked = camArray;
    masked(~inside) = -Inf;
    [~, maxIdx] = max(masked(:));
    [py, px] = ind2sub(size(camArray), maxIdx);

    if ~isempty(anatomy)
        region = dr.quadrantLabel(px, py, anatomy.disc, anatomy.nasalSide);
    elseif py < size(camArray,1)*0.5
        region = "superior frame";
    else
        region = "inferior frame";
    end

    sorted = sort(v, "descend");
    topN = max(1, round(numel(sorted)*0.1));
    total = sum(v);
    if total > 1e-8
        concentration = sum(sorted(1:topN)) / total;
    else
        concentration = 0;
    end

    C = dr.Config();
    d = struct("region", region, ...
               "focal", concentration > 0.35, ...
               "flat", cam.stats.median >= C.cam.flatMedian, ...
               "concentration", concentration, ...
               "median", cam.stats.median, ...
               "peakX", px, "peakY", py);
end
