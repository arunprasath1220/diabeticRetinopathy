function label = quadrantLabel(px, py, disc, nasalSide)
%QUADRANTLABEL  Name the retinal quadrant a point falls in.
%
%   L = dr.quadrantLabel(PX, PY, DISC, NASALSIDE) returns one of
%   "Superior-Nasal", "Superior-Temporal", "Inferior-Nasal", "Inferior-Temporal".
%
%   The axes run through the optic disc, which is what the ICDR 4-2-1 rule counts
%   hemorrhages against. Nasal and temporal are assigned from which side of the
%   disc the fovea lies on, so the labelling is correct for a right or a left eye
%   without ever being told which it is.

    if py < disc.y
        vertical = "Superior";
    else
        vertical = "Inferior";
    end

    leftOfDisc = px < disc.x;
    if nasalSide == "left"
        if leftOfDisc
            horizontal = "Nasal";
        else
            horizontal = "Temporal";
        end
    else
        if leftOfDisc
            horizontal = "Temporal";
        else
            horizontal = "Nasal";
        end
    end

    label = vertical + "-" + horizontal;
end
