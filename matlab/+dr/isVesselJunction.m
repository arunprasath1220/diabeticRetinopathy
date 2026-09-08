function tf = isVesselJunction(ctx)
%ISVESSELJUNCTION  Is this a crossing, a bifurcation or a bend, rather than a lesion?
%
%   Read from what leaves the region (see DR.VESSELARMS):
%
%     0 or 1 arm        free-standing lesion, or one sitting against a vessel
%     2 opposite arms   a vessel passing straight through - a microaneurysm or
%                       dot hemorrhage on a vessel looks like this, and those are
%                       real, so this case is KEPT
%     2 arms at an angle a bend: the vessel turns here, and the turn is what was
%                       detected
%     3 or more arms    a crossing or a bifurcation
%
%   Keeping the antipodal case is what stops this test from throwing away genuine
%   lesions on vessels, which is the failure mode that matters most here.

    C = dr.Config();

    if ctx.coverage >= C.lesion.junctionRingSaturation
        tf = true;
        return;
    end
    if ctx.arms >= 3
        tf = true;
        return;
    end
    if ctx.arms == 2
        tf = ~dr.isThroughVessel(ctx);
        return;
    end
    tf = false;
end
