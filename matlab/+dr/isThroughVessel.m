function tf = isThroughVessel(ctx)
%ISTHROUGHVESSEL  Two arms leaving in roughly opposite directions.
%
%   A vessel runs *through* this region rather than ending, branching or turning
%   in it. That case is kept, not rejected: a microaneurysm beside a venule looks
%   exactly like this, and it is a real finding and by far the commonest early
%   sign of diabetic retinopathy. Deleting the class would cost more than it
%   saves.

    C = dr.Config();
    if ctx.arms ~= 2
        tf = false;
        return;
    end
    d = abs(ctx.angles(1) - ctx.angles(2));
    if d > 180
        d = 360 - d;
    end
    tf = abs(d - 180) <= C.lesion.junctionAntipodalTolDeg;
end
