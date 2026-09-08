function panels = renderOverlays(I, anatomy, lesions, cam)
%RENDEROVERLAYS  Build the annotated panels the report and the UI display.
%
%   PANELS = dr.renderOverlays(I, ANATOMY, LESIONS, CAM) returns a struct of RGB
%   images: anatomy, lesionOverlay, binaryMask, typedMask and, if CAM was given,
%   camColour and camRegions.
%
%   Drawing is done with INSERTSHAPE / INSERTTEXT / INSERTMARKER rather than by
%   plotting to a figure and capturing it, so the annotations land on exact pixel
%   coordinates and the result is reproducible headlessly - which matters when
%   the same code has to run inside a Simulink deployment simulation with no
%   display attached.
%
%   Every marker shape is distinct per lesion type. Labelling only the largest
%   few left most marks anonymous, which made the overlay impossible to read.

    C = dr.Config();
    [h, w, ~] = size(I);
    panels = struct();

    % ---- Anatomy --------------------------------------------------------
    A = im2uint8(I);
    if ~isempty(anatomy)
        d = anatomy.disc;
        f = anatomy.fovea;
        A = insertShape(A, "circle", [d.x d.y d.radius], ...
                        "Color", "white", "LineWidth", 2);
        A = insertShape(A, "circle", [f.x f.y anatomy.maculaRadius], ...
                        "Color", "cyan", "LineWidth", 1);
        A = insertMarker(A, [f.x f.y], "plus", "Color", "cyan", "Size", 8);
        % Quadrant axes through the disc, which is what the ICDR rule counts against
        A = insertShape(A, "line", [0 d.y w d.y], "Color", "white", "LineWidth", 1);
        A = insertShape(A, "line", [d.x 0 d.x h], "Color", "white", "LineWidth", 1);
        A = insertText(A, [d.x d.y-d.radius-4], "Optic disc", ...
                       "AnchorPoint", "CenterBottom", "BoxOpacity", 0.5, ...
                       "TextColor", "white");
        A = insertText(A, [f.x f.y-anatomy.maculaRadius-4], "Macula", ...
                       "AnchorPoint", "CenterBottom", "BoxOpacity", 0.5, ...
                       "TextColor", "cyan");
    end
    panels.anatomy = A;

    % ---- Lesion overlay and masks ---------------------------------------
    O = im2uint8(I);
    binary = false(h, w);
    typed  = zeros(h, w, "uint8");

    colours = struct("MA","yellow", "HEM","red", "HE","green", "CWS","magenta");

    for i = 1:numel(lesions.candidates)
        c = lesions.candidates(i);
        r = max(4, sqrt(c.area/pi)*1.6);
        col = colours.(c.type);

        switch c.type
            case {"MA", "HEM"}
                O = insertShape(O, "circle", [c.cx c.cy r], ...
                                "Color", col, "LineWidth", 2);
                if c.type == "HEM"
                    O = insertShape(O, "circle", [c.cx c.cy r*1.6], ...
                                    "Color", col, "LineWidth", 1);
                end
            otherwise
                O = insertShape(O, "rectangle", [c.cx-r c.cy-r 2*r 2*r], ...
                                "Color", col, "LineWidth", 2);
        end

        % A candidate lying along the course of a vessel is kept - a
        % microaneurysm beside a venule is real and common - but struck through,
        % because a wide spot in a vessel looks exactly the same.
        if c.onVessel
            O = insertShape(O, "line", [c.cx-r c.cy+r c.cx+r c.cy-r], ...
                            "Color", "white", "LineWidth", 1);
        end

        binary(c.pixels) = true;
        typed(c.pixels) = C.lesionTypes.(c.type).grey;
    end

    panels.lesionOverlay = O;
    panels.binaryMask = im2uint8(binary);
    panels.typedMask = typed;

    % ---- Grad-CAM -------------------------------------------------------
    if nargin >= 4 && ~isempty(cam)
        % Feathered alpha, so the analysed-field boundary fades rather than
        % stepping. See DR.CONFINECAMTORETINA.
        heat = im2uint8(ind2rgb(uint8(cam.camArray*255), turbo(256)));
        alpha = 0.45 * cam.feather;
        base = im2double(I);
        blend = base .* (1-alpha) + im2double(heat) .* alpha;
        Cimg = im2uint8(blend);

        Cimg = insertShape(Cimg, "circle", ...
                    [cam.field.cx cam.field.cy cam.field.r], ...
                    "Color", "white", "LineWidth", 1);
        panels.camColour = Cimg;

        blobs = dr.findDefectBlobs(cam, anatomy);
        R = im2uint8(I);
        num = 0;
        for k = 1:numel(blobs)
            b = blobs(k);
            rr = max(9, b.radius);
            if b.onDisc
                % Drawn but not numbered: expected attention, not a finding.
                R = insertShape(R, "circle", [b.cx b.cy rr], ...
                                "Color", "white", "LineWidth", 1);
                R = insertText(R, [b.cx b.cy-rr-4], "disc", ...
                               "AnchorPoint", "CenterBottom", "BoxOpacity", 0.5);
            else
                num = num + 1;
                R = insertShape(R, "circle", [b.cx b.cy rr], ...
                                "Color", "yellow", "LineWidth", 2);
                R = insertText(R, [b.cx b.cy-rr-4], string(num), ...
                               "AnchorPoint", "CenterBottom", "BoxOpacity", 0.6);
            end
        end
        panels.camRegions = R;
        panels.camBlobs = blobs;
    end
end
