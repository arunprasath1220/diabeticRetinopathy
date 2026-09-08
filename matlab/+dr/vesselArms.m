function ctx = vesselArms(cx, cy, area, vesselZone, armLength)
%VESSELARMS  What runs out of a candidate region, and in which directions.
%
%   CTX = dr.vesselArms(CX, CY, AREA, VESSELZONE, ARMLENGTH) samples a ring just
%   outside the region and returns CTX.arms, CTX.coverage and CTX.angles.
%
%   This exists because the directional top-hat has one systematic blind spot,
%   and it is the whole reason a healthy retina once came back covered in
%   microaneurysm marks. The roundness test asks whether a structure is filled in
%   by a linear closing in *every* direction. A vessel is not - it survives the
%   element lying along it. But that argument only holds where a vessel is
%   locally straight and alone. Where two vessels cross, where one branches, and
%   at the apex of a tight bend, there is no direction in which the structure is
%   straight, so every orientation fills it and it seeds exactly as a lesion
%   does. Those three configurations lie all over the vascular arcades.
%
%   No refinement of the top-hat can separate them, because at the junction the
%   two really do look the same. What differs is the surroundings: a lesion is an
%   isolated blob with normal retina around it, a junction has vessel running out
%   of it in three or four directions. So the region is judged by what leaves it.
%
%   Contiguous runs of hit directions are merged, so one vessel several
%   directions wide counts as one arm rather than several.

    C = dr.Config();
    L = C.lesion;
    [h, w] = size(vesselZone);

    r0 = sqrt(max(area,1)/pi);
    inner = r0 + L.armCollarPad;
    outer = inner + max(8, armLength);
    steps = max(6, round(outer - inner));

    nd = L.armDirections;
    hit = false(1, nd);

    for d = 0:nd-1
        th = 2*pi*d/nd;
        cth = cos(th);  sth = sin(th);
        % A perpendicular tolerance of one pixel keeps the ray on a vessel that
        % leans slightly, without letting it wander onto a neighbouring one.
        px = -sth;  py = cth;

        on = 0;  seen = 0;
        for k = 0:steps
            r = inner + (outer-inner)*k/steps;
            any_ = false;
            for o = -1:1
                x = round(cx + r*cth + o*px);
                y = round(cy + r*sth + o*py);
                if x < 1 || y < 1 || x > w || y > h
                    continue;
                end
                if vesselZone(y, x)
                    any_ = true;
                    break;
                end
            end
            seen = seen + 1;
            on = on + any_;
        end

        if seen > 0 && on/seen >= L.armPersistence
            hit(d+1) = true;
        end
    end

    hits = nnz(hit);
    if hits == 0
        ctx = struct("arms", 0, "coverage", 0, "angles", []);
        return;
    end
    if hits == nd
        ctx = struct("arms", 1, "coverage", 1, "angles", 0);
        return;
    end

    % Begin at a gap so that runs do not wrap around the end of the array.
    start = find(~hit, 1) - 1;              % 0-based offset
    angles = [];
    i = 0;
    while i < nd
        j = mod(start + i, nd) + 1;
        if ~hit(j)
            i = i + 1;
            continue;
        end
        len = 0;  sum_ = 0;
        while len < nd
            jj = mod(start + i + len, nd) + 1;
            if ~hit(jj)
                break;
            end
            sum_ = sum_ + (start + i + len);
            len = len + 1;
        end
        angles(end+1) = mod(sum_/len, nd) * 360/nd; %#ok<AGROW>
        i = i + len;
    end

    ctx = struct("arms", numel(angles), "coverage", hits/nd, "angles", angles);
end
