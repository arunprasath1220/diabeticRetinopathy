function blobs = findDefectBlobs(cam, anatomy)
%FINDDEFECTBLOBS  Discrete regions of concentrated model attention.
%
%   BLOBS = dr.findDefectBlobs(CAM, ANATOMY) returns a struct array with cx, cy,
%   area, peak, radius and quadrant, strongest first.
%
%   Two things decide the threshold, and the second is why an image with nothing
%   to find now produces no circles at all. Grad-CAM is scaled to its own
%   maximum, so some pixel is always 1.0 and a fixed fraction-of-peak cut always
%   returns a region - on a healthy retina, the tallest bump in a flat map.
%   Requiring the region also to stand clear of the retina's own typical
%   activation removes exactly that case, and leaves a genuine hotspot, which
%   clears both cuts easily, untouched.
%
%   A map that is warm everywhere localises nothing whatever its peak, so a high
%   median suppresses marking entirely rather than producing a region the size of
%   the retina.

    C = dr.Config();
    camArray = cam.camArray;
    inside = cam.inside;

    % Field order here must match the order B is built in below, or the
    % growing assignment fails. Declared once, so there is one place to look.
    blobs = struct("cx",{}, "cy",{}, "area",{}, "peak",{}, "radius",{}, ...
                   "onDisc",{}, "quadrant",{});

    if cam.stats.median >= C.cam.flatMedian
        return;                                  % warm everywhere: no localisation
    end

    noiseCut = cam.stats.median + C.cam.noiseK * cam.stats.mad;
    cut = max(C.cam.blobThreshold, noiseCut);

    bw = inside & (camArray >= cut);
    minArea = max(20, round(numel(camArray) * C.cam.minAreaFrac));
    bw = bwareaopen(bw, minArea, 4);

    cc = bwconncomp(bw, 4);
    stats = regionprops(cc, "Area", "Centroid");

    for k = 1:cc.NumObjects
        b.cx = stats(k).Centroid(1);
        b.cy = stats(k).Centroid(2);
        b.area = stats(k).Area;
        b.peak = max(camArray(cc.PixelIdxList{k}));
        % Area-equivalent radius, widened a little so the marker encloses rather
        % than bisects the region it is drawn around.
        b.radius = sqrt(b.area/pi) * 1.25;

        % The optic disc draws strong attention from almost any fundus
        % classifier: it is the brightest, most distinctive structure in the
        % frame and the network uses it to orient itself. That attention is real
        % and worth showing, but it is not a finding.
        if ~isempty(anatomy)
            b.onDisc = hypot(b.cx - anatomy.disc.x, b.cy - anatomy.disc.y) ...
                       <= anatomy.disc.radius*1.2;
            b.quadrant = dr.quadrantLabel(b.cx, b.cy, anatomy.disc, anatomy.nasalSide);
        else
            b.onDisc = false;
            b.quadrant = "n/a";
        end
        blobs(end+1) = b; %#ok<AGROW>
    end

    if ~isempty(blobs)
        [~, ord] = sortrows([[blobs.peak]' [blobs.area]'], [-1 -2]);
        blobs = blobs(ord);
        if numel(blobs) > C.cam.regionLimit
            blobs = blobs(1:C.cam.regionLimit);
        end
    end
end
