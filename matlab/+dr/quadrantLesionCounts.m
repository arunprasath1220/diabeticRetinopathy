function grid = quadrantLesionCounts(lesions, anatomy)
%QUADRANTLESIONCOUNTS  Candidate counts per type per retinal quadrant.
%
%   GRID = dr.quadrantLesionCounts(LESIONS, ANATOMY) returns a struct indexed
%   GRID.(type).(quadrant), or [] when the landmarks are unavailable - because
%   without a disc position there are no quadrant axes, and inventing them would
%   put every count in the wrong cell.
%
%   Laid out this way because the ICDR severity-3 rule counts hemorrhages by
%   quadrant. It is not itself a 4-2-1 assessment; see DR.ASSESSSEVERITY for why
%   the distinction matters.

    grid = [];
    if isempty(anatomy) || isempty(lesions)
        return;
    end

    C = dr.Config();
    quads = C.icdr.quadrantNames;

    for t = C.lesionOrder
        for q = quads
            grid.(t).(matlab.lang.makeValidName(q)) = 0;
        end
    end

    for i = 1:numel(lesions.candidates)
        c = lesions.candidates(i);
        q = dr.quadrantLabel(c.cx, c.cy, anatomy.disc, anatomy.nasalSide);
        f = matlab.lang.makeValidName(q);
        if isfield(grid, c.type) && isfield(grid.(c.type), f)
            grid.(c.type).(f) = grid.(c.type).(f) + 1;
        end
    end
end
