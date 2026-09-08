function ch = imageChannels(I)
%IMAGECHANNELS  Pull the channels the pipeline reasons about out of an RGB image.
%
%   CH = dr.imageChannels(I) takes an H-by-W-by-3 image of any numeric class and
%   returns a struct of double planes on the 0..255 scale:
%
%     ch.lum    Rec.601 luminance, via RGB2GRAY
%     ch.green  the green plane, which carries the strongest blood contrast and
%               is what every dark-lesion test is measured on
%     ch.red    ) carried only so that hue can be read. Nothing in the
%     ch.blue   ) morphology uses them, but telling lipid exudate from optic
%               ) disc tissue is a question about colour, not about shape.
%     ch.h, ch.w, ch.n
%
%   Luminance is computed through RGB2GRAY on a double image rather than on the
%   integer original: on uint8 input RGB2GRAY rounds to uint8, and the pipeline's
%   deviation maps are thresholded at single grey levels, where that rounding is
%   not negligible.

    if ndims(I) ~= 3 || size(I,3) < 3
        error("dr:imageChannels:notRGB", "expected an H-by-W-by-3 RGB image");
    end

    D = im2double(I);                       % 0..1, any input class
    ch.red   = D(:,:,1) * 255;
    ch.green = D(:,:,2) * 255;
    ch.blue  = D(:,:,3) * 255;
    ch.lum   = rgb2gray(D) * 255;           % 0.299R + 0.587G + 0.114B

    [ch.h, ch.w, ~] = size(I);
    ch.n = ch.h * ch.w;
end
