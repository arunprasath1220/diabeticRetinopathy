function cam = computeGradCAM(net, I, classIndex)
%COMPUTEGRADCAM  Step 3: where the classifier looked, confined to the retina.
%
%   CAM = dr.computeGradCAM(NET, I, CLASSINDEX) returns a struct with:
%     camArray   normalised 0..1 map at image resolution, zero outside the retina
%     inside     the hard analysis mask
%     feather    a soft copy of the same boundary, for drawing only
%     field      the circle that boundary lies on, so panels can draw it
%     stats      median and robust spread over retina only
%
%   The map is always computed for the "DR present" class regardless of what the
%   network predicted. An explanation of why the model said "no disease" is not
%   what a clinician needs; what they need is where the evidence for disease
%   would have been.
%
%   ---- Why the map is confined ----
%   A Grad-CAM cell that straddles the aperture edge is computed from a receptive
%   field that is mostly black surround, so its value says nothing about retina -
%   but it is often large, and it lands in a ring around the image where it looks
%   like a finding. The map is therefore masked to the retina eroded by half a
%   CAM cell, and the boundary is drawn rather than left implicit, so that an
%   otherwise unexplained ring becomes a stated exclusion.
%
%   Every statistic is taken over retina only. Measured over the whole frame the
%   black surround contributes a large block of zeros that inflates the
%   concentration figure - a map spread evenly across the retina still looks
%   focal simply because the retina is a minority of the frame - and the peak it
%   reports can land outside the eye entirely.

    C = dr.Config();
    x = dr.prepareInput(I);

    featureLayer = pickFeatureLayer(net, C);

    % GRADCAM is the Deep Learning Toolbox implementation; it handles the
    % reduction layer and the gradient plumbing.
    raw = gradCAM(net, x, classIndex, "FeatureLayer", featureLayer);

    [h, w, ~] = size(I);
    raw = imresize(double(raw), [h w], "bilinear");
    raw = max(0, raw);

    cam = dr.confineCamToRetina(I, raw);
    cam.featureLayer = featureLayer;
    cam.classIndex = classIndex;
end

function name = pickFeatureLayer(net, C)
%PICKFEATURELAYER  Deepest listed convolutional layer that this network has.
%
%   The very last convolutional layer gives the sharpest class discrimination but
%   the coarsest spatial grid, and on a 224-pixel input that grid is 7x7 - about
%   32 pixels per cell, which is wider than many of the lesions being explained.
%   One stage earlier doubles the resolution for a modest loss of specificity,
%   which is the better trade when the output is a localisation claim.

    names = string({net.Layers.Name});
    for cand = C.model.featureLayerCandidates
        if any(names == cand)
            name = cand;
            return;
        end
    end

    % Fall back to the last layer that produces a spatial activation.
    isConv = arrayfun(@(l) isa(l, "nnet.cnn.layer.Convolution2DLayer") || ...
                           isa(l, "nnet.cnn.layer.ReLULayer"), net.Layers);
    idx = find(isConv, 1, "last");
    if isempty(idx)
        error("dr:computeGradCAM:noLayer", ...
              "no usable convolutional layer for Grad-CAM");
    end
    name = names(idx);
end
