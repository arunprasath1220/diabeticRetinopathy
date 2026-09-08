function out = enhanceImage(I)
%ENHANCEIMAGE  Step 1b: denoise, flatten illumination, restore local contrast.
%
%   OUT = dr.enhanceImage(I) works on luminance alone and puts the original
%   chroma back afterwards, so nothing here can shift the colour of a lesion.
%   That matters downstream: the peripapillary test in DR.DETECTLESIONCANDIDATES
%   separates yellow exudate from white disc tissue by hue, and an enhancement
%   that touched chroma would quietly break it.
%
%   Three stages, in this order:
%
%     1. A light denoise - 30% of a radius-1 box blur - so that the illumination
%        estimate below is not fitted to sensor noise.
%     2. Illumination flattening. Subtracting a heavily blurred copy and adding
%        the global mean back removes the vignette and the uneven flash falloff
%        that every handheld fundus camera produces, without changing the mean
%        level the later thresholds are calibrated against.
%     3. CLAHE, which is what makes faint lesions visible again after the
%        flattening has compressed the range.
%
%   Note that RGB2YCBCR is deliberately *not* used. It implements the studio-
%   swing BT.601 convention, where luma occupies 16..235; the pipeline's
%   thresholds are all stated on a full-range 0..255 scale, and running the
%   conversion through the studio-swing form would silently rescale every one of
%   them. The full-range JFIF form is written out instead.

    C = dr.Config();

    D = double(I);
    R = D(:,:,1);  G = D(:,:,2);  B = D(:,:,3);

    Y  = 0.299*R + 0.587*G + 0.114*B;
    Cb = -0.168736*R - 0.331264*G + 0.5*B + 128;
    Cr = 0.5*R - 0.418688*G - 0.081312*B + 128;

    % ---- 1. light denoise ----------------------------------------------
    mix = C.enhance.denoiseMix;
    Y = (1-mix)*Y + mix*dr.boxBlur(Y, 1);

    % ---- 2. illumination flattening ------------------------------------
    [h, w] = size(Y);
    radius = max(4, round(min(w, h) / 10));
    Y = min(255, max(0, Y - dr.boxBlur(Y, radius) + mean(Y(:))));

    % ---- 3. CLAHE -------------------------------------------------------
    Y = adapthisteq(Y/255, ...
            "NumTiles",     C.enhance.tiles, ...
            "ClipLimit",    C.enhance.clipLimit, ...
            "NBins",        C.enhance.numBins, ...
            "Range",        "full", ...
            "Distribution", "uniform") * 255;

    % ---- back to RGB, chroma untouched ---------------------------------
    outR = Y + 1.402*(Cr - 128);
    outG = Y - 0.344136*(Cb - 128) - 0.714136*(Cr - 128);
    outB = Y + 1.772*(Cb - 128);

    out = cat(3, outR, outG, outB);
    out = uint8(min(255, max(0, out)));
end
