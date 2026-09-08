function x = prepareInput(I)
%PREPAREINPUT  Put an image into the form the classifier was trained on.
%
%   X = dr.prepareInput(I) resizes to the network input size and scales to
%   [-1, 1], which is the MobileNet convention.
%
%   Kept in one function because it is called from both DR.CLASSIFYIMAGE and
%   DR.COMPUTEGRADCAM. If the two ever disagreed, the heatmap would be explaining
%   a slightly different image from the one that was graded - an explanation that
%   is wrong in a way nobody would notice by looking at it.

    C = dr.Config();
    resized = imresize(I, C.model.inputSize);
    x = single(resized);
    x = (x - 127.5) / 127.5;
end
