function out = hysteresisMask(strong, weak)
%HYSTERESISMASK  Flood a confident mask outward through a permissive one.
%
%   OUT = dr.hysteresisMask(STRONG, WEAK) keeps every pixel of WEAK that is
%   8-connected to a pixel of STRONG, plus STRONG itself.
%
%   This is morphological reconstruction, and IMRECONSTRUCT does it in one call.
%
%   Both masks are already gated on the same evidence, so this only ever connects
%   what the permissive threshold already believed. What it adds is the
%   requirement that a weak pixel be *attached* to a confident one, which is
%   exactly what distinguishes the faint continuation of a real vessel from an
%   isolated patch of texture at the same amplitude. It is what turns a vessel
%   map of disconnected fragments into something shaped like a tree.
%
%   Eight-connectivity, because a vessel crossing the pixel grid at an angle is a
%   staircase and four-connectivity breaks it into beads.

    strong = logical(strong);
    weak   = logical(weak);

    % The marker must lie inside the mask for reconstruction to be defined; any
    % strong pixel outside the permissive mask is then added back, so a seed can
    % never be lost to a threshold that was meant to be more forgiving than it.
    out = imreconstruct(strong & weak, weak, 8) | strong;
end
