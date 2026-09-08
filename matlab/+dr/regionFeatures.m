function comps = regionFeatures(bw, contrastMap, gradientMap, seedMask)
%REGIONFEATURES  Connected components with the features the type rules need.
%
%   COMPS = dr.regionFeatures(BW, CONTRASTMAP, GRADIENTMAP, SEEDMASK) returns a
%   struct array, one entry per 4-connected component of BW, with fields:
%
%     pixels        linear indices of the component
%     area          pixel count
%     cx, cy        centroid, 1-based
%     bw, bh        bounding box width and height
%     fillRatio     area / (bw*bh) - how solidly the box is filled
%     aspect        long side / short side of the box
%     maxContrast   strongest deviation anywhere in the region
%     meanContrast  mean deviation
%     meanGradient  mean edge strength, which separates hard exudate (sharp) from
%                   cotton wool spot (soft)
%     hasSeed       whether any pixel carries directional evidence
%
%   BWCONNCOMP and REGIONPROPS do the labelling and the geometry. The remaining
%   features are per-pixel statistics over PixelIdxList, which is why the pixel
%   list is kept rather than discarded.
%
%   Four-connectivity, not eight. These are compact blobs, and eight-connectivity
%   bridges two lesions that merely touch at a corner into one region, which
%   changes both the count and the shape tests applied to it.

    if nargin < 4
        seedMask = [];
    end

    cc = bwconncomp(logical(bw), 4);
    stats = regionprops(cc, "Area", "Centroid", "BoundingBox");

    n = cc.NumObjects;
    comps = struct("pixels", {}, "area", {}, "cx", {}, "cy", {}, ...
                   "bw", {}, "bh", {}, "fillRatio", {}, "aspect", {}, ...
                   "maxContrast", {}, "meanContrast", {}, "meanGradient", {}, ...
                   "hasSeed", {});

    for k = 1:n
        idx = cc.PixelIdxList{k};
        bb  = stats(k).BoundingBox;         % [x y width height], corner-based
        bwid = bb(3);
        bhei = bb(4);

        c.pixels = idx;
        c.area   = stats(k).Area;
        c.cx     = stats(k).Centroid(1);
        c.cy     = stats(k).Centroid(2);
        c.bw     = bwid;
        c.bh     = bhei;
        c.fillRatio = c.area / max(1, bwid*bhei);
        c.aspect    = max(bwid,bhei) / max(1, min(bwid,bhei));

        cvals = contrastMap(idx);
        c.maxContrast  = max(cvals);
        c.meanContrast = mean(cvals);
        c.meanGradient = mean(gradientMap(idx));

        if isempty(seedMask)
            c.hasSeed = true;               % no seeding requirement asked for
        else
            c.hasSeed = any(seedMask(idx));
        end

        comps(end+1) = c; %#ok<AGROW>
    end
end
