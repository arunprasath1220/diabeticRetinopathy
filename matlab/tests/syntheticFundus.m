function I = syntheticFundus(opts)
%SYNTHETICFUNDUS  Build a fundus image with known ground truth.
%
%   I = syntheticFundus() returns a plain retina with a disc and two vessels.
%   I = syntheticFundus(Name=Value) adds the structures a test needs.
%
%   Options:
%     Size        [H W], default [600 600]
%     Disc        struct with x, y, r
%     Crescent    struct with x, y, r, from, to - a scleral arc outside the rim
%     Spots       struct array with x, y, r, colour - arbitrary marks
%     Exudates    struct array with x, y, r - yellow lipid marks
%     Vessels     logical, default true
%
%   This exists because the repository holds no real fundus images, and a
%   detector tuned only against images you cannot redistribute is a detector
%   nobody else can check. A synthetic scene is weaker evidence than a real
%   retina - the noise is wrong, the texture is wrong, the vessels are too
%   regular - but it does establish that a specific rule fires on a specific
%   structure, which is what the tests here assert and all they claim.
%
%   The colours matter and are not arbitrary. Retinal background is red-orange;
%   the disc is pale and desaturated; exudate is yellow; sclera and reflex are
%   white-grey. The peripapillary colour test in DR.DETECTLESIONCANDIDATES turns
%   on exactly that difference, so a test image that got it wrong would pass a
%   broken detector.

    arguments
        opts.Size (1,2) double = [600 600]
        opts.Disc = struct("x", 200, "y", 300, "r", 50)
        opts.Crescent = []
        opts.Spots = []
        opts.Exudates = []
        opts.Vessels (1,1) logical = true
    end

    H = opts.Size(1);  W = opts.Size(2);
    rng(7);                                     % reproducible noise

    [X, Y] = meshgrid(1:W, 1:H);
    cx = W/2;  cy = H/2;  R = min(W,H)*0.47;
    d = hypot(X-cx, Y-cy);

    % ---- Background retina, vignetted ----------------------------------
    vig = 1 - 0.30*(d/R).^3;
    noise = (rand(H,W)-0.5)*6;
    I = zeros(H, W, 3);
    I(:,:,1) = 178*vig + noise;
    I(:,:,2) =  82*vig + noise;
    I(:,:,3) =  44*vig + noise;
    outside = d > R;
    for k = 1:3
        ch = I(:,:,k);  ch(outside) = 0;  I(:,:,k) = ch;
    end

    % ---- Optic disc: pale and DESATURATED, not merely bright ------------
    if ~isempty(opts.Disc)
        I = blob(I, X, Y, opts.Disc.x, opts.Disc.y, opts.Disc.r, 8, [236 226 216]);
    end

    % ---- Scleral crescent: the classic false-exudate source -------------
    if ~isempty(opts.Crescent)
        c = opts.Crescent;
        I = arc(I, X, Y, c.x, c.y, c.r, 5, [214 202 196], c.from, c.to);
    end

    % ---- Arbitrary marks -------------------------------------------------
    if ~isempty(opts.Spots)
        for k = 1:numel(opts.Spots)
            s = opts.Spots(k);
            I = blob(I, X, Y, s.x, s.y, s.r, 3, s.colour);
        end
    end

    % ---- Exudates: YELLOW, which is what makes them separable ------------
    if ~isempty(opts.Exudates)
        for k = 1:numel(opts.Exudates)
            e = opts.Exudates(k);
            I = blob(I, X, Y, e.x, e.y, e.r, 2, [246 228 108]);
        end
    end

    % ---- A couple of vessels leaving the disc ----------------------------
    if opts.Vessels && ~isempty(opts.Disc)
        for t = 0:419
            for s = [1 -1]
                ang = s*0.55;
                px = round(opts.Disc.x + t*cos(ang));
                py = round(opts.Disc.y + t*sin(ang));
                for o = -2:2
                    yy = py + o;
                    if px < 1 || px > W || yy < 1 || yy > H, continue; end
                    a = 0.5;  if o == 0, a = 0.9; end
                    I(yy,px,1) = I(yy,px,1)*(1-a) + 96*a;
                    I(yy,px,2) = I(yy,px,2)*(1-a) + 26*a;
                    I(yy,px,3) = I(yy,px,3)*(1-a) + 22*a;
                end
            end
        end
    end

    I = uint8(min(255, max(0, I)));
end

function I = blob(I, X, Y, bx, by, r, soft, colour)
%BLOB  A soft-edged disc: full strength inside r, falling off over `soft`.
    d = hypot(X-bx, Y-by);
    a = zeros(size(d));
    a(d <= r) = 1;
    ring = d > r & d <= r+soft;
    a(ring) = 1 - (d(ring)-r)/soft;
    for k = 1:3
        ch = I(:,:,k);
        ch = ch.*(1-a) + colour(k)*a;
        I(:,:,k) = ch;
    end
end

function I = arc(I, X, Y, bx, by, r, soft, colour, angFrom, angTo)
%ARC  A hollow crescent, for peripapillary atrophy.
    d = hypot(X-bx, Y-by);
    th = atan2(Y-by, X-bx);
    th(th < 0) = th(th < 0) + 2*pi;

    a = zeros(size(d));
    band = d >= r*0.92 & d <= r+soft & th >= angFrom & th <= angTo;
    a(band) = 1;
    fade = band & d > r;
    a(fade) = 1 - (d(fade)-r)/soft;
    a = max(0, a);

    for k = 1:3
        ch = I(:,:,k);
        ch = ch.*(1-a) + colour(k)*a;
        I(:,:,k) = ch;
    end
end
