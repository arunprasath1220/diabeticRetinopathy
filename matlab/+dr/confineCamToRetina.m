function cam = confineCamToRetina(I, raw)
%CONFINECAMTORETINA  Mask, normalise and characterise a raw activation map.
%
%   CAM = dr.confineCamToRetina(I, RAW)
%
%   The hard mask and the soft one are deliberately not the same thing. Painting
%   an overlay through the hard mask leaves a step at its edge - analysed retina
%   tinted, the trimmed rim not - and that step reads as a rendering fault rather
%   than as the edge of the analysed field. A box-blurred copy of the mask gives
%   a weight that falls from one to zero across a few pixels, so the overlay
%   fades out instead. Detection still uses the hard mask: what is measured and
%   what is painted are different questions.

    C = dr.Config();
    ch = dr.imageChannels(I);
    w = ch.w;  h = ch.h;

    ap = dr.apertureGeometry(ch.lum);
    inside = dr.retinaFieldMask(ch.lum, 0.05);

    % Half a Grad-CAM cell at this display size. Below the CAM's own resolution
    % there is nothing to gain from a finer trim, and above it genuine peripheral
    % evidence starts being discarded.
    bleed = max(2, round(min(w,h) / (C.cam.gridHint*2)));
    inside = dr.erodeMask(inside, bleed);

    feather = dr.boxBlur(double(inside), ...
                min(24, max(3, round(bleed * C.cam.featherFrac))));

    field = struct("cx", ap.cx, "cy", ap.cy, ...
                   "r", max(1, ap.R*0.955 - bleed));

    insideCount = nnz(inside);
    rawMax = max(raw(inside));
    if isempty(rawMax), rawMax = 0; end

    camArray = zeros(h, w);
    if insideCount == 0 || rawMax <= 1e-12
        cam = struct("camArray", camArray, "inside", inside, ...
                     "feather", feather, "field", field, ...
                     "insideCount", insideCount, "rawMax", rawMax, ...
                     "h", h, "w", w, ...
                     "stats", struct("median",0, "mad",0));
        return;
    end

    camArray(inside) = raw(inside) / rawMax;

    v = camArray(inside);
    stats = struct("median", median(v), "mad", 1.4826*mad(v,1));

    cam = struct("camArray", camArray, "inside", inside, ...
                 "feather", feather, "field", field, ...
                 "insideCount", insideCount, "rawMax", rawMax, ...
                 "h", h, "w", w, "stats", stats);
end
