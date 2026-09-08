function m = computeMetrics(I)
%COMPUTEMETRICS  Step 1: measure the things that decide whether to proceed.
%
%   M = dr.computeMetrics(I) returns focusVar, brightMean, brightStd and fov.
%
%   Everything is measured on a copy resized to a fixed edge (see DR.CONFIG), so
%   the thresholds in DR.ASSESSQUALITY mean the same thing regardless of what the
%   camera's native resolution happens to be. A focus score computed at native
%   resolution is not comparable between a 6 MP handheld and a 1 MP phone adapter
%   and would gate the two differently for no clinical reason.

    C = dr.Config();

    scale = min(1, C.metricDim / max(size(I,1), size(I,2)));
    small = imresize(I, scale, "bilinear");
    gray  = rgb2gray(im2double(small)) * 255;

    % ---- Focus: variance of the Laplacian ------------------------------
    % A sharp image has strong second derivatives; a blurred one does not. The
    % variance rather than the mean, because the mean of a Laplacian is near zero
    % on any image and says nothing.
    K = [0 1 0; 1 -4 1; 0 1 0];
    lap = imfilter(gray, K);
    interior = lap(2:end-1, 2:end-1);        % the border is padding, not signal
    m.focusVar = var(interior(:), 1);        % population variance

    % ---- Exposure ------------------------------------------------------
    m.brightMean = mean(gray(:));
    m.brightStd  = std(gray(:), 1);

    % ---- Field of view: how much of the frame is not black border ------
    m.fov = nnz(gray > 15) / numel(gray);
end
