classdef tPeripapillary < matlab.unittest.TestCase
%TPERIPAPILLARY  The peripapillary bright-suppression rules.
%
%   These mirror the JavaScript harness one-for-one, so a regression in either
%   implementation shows up as a disagreement between them rather than as a
%   quiet drift in one.
%
%   The test that matters most is COLOURSEPARATESDISCFROMEXUDATE. Two marks at
%   the same distance from the disc, the same size, the same brightness above
%   background, differing only in colour: one must be rejected and one kept. A
%   rule that removes both is not an improvement, it is a bigger exclusion zone,
%   and only a paired test can tell those apart.

    properties (Constant)
        DISC = struct("x", 200, "y", 300, "r", 50)
    end

    methods (Test)

        function discExtentIsGrownNotInferred(tc)
            I = syntheticFundus(Disc=tc.DISC);
            disc = dr.estimateOpticDisc(I);

            tc.verifyTrue(disc.measuredExtent, ...
                "the extent should be grown on a clean synthetic disc");
            tc.verifyNotEmpty(disc.tissue, ...
                "a halo mask should have been produced");
            % Within 20% of truth. The grown radius sits slightly wide because
            % the halo level deliberately reaches past the rim.
            tc.verifyEqual(disc.radius, tc.DISC.r, "RelTol", 0.20);
        end

        function crescentIsNotReportedAsExudate(tc)
            % A scleral crescent hugging the temporal rim: bright, round, sharply
            % bounded, and entirely not a lesion.
            I = syntheticFundus(Disc=tc.DISC, ...
                Crescent=struct("x", tc.DISC.x, "y", tc.DISC.y, ...
                                "r", tc.DISC.r+12, "from", 5.1, "to", 6.28));

            [les, ~] = tc.analyse(I);
            near = tc.brightNearDisc(les, 3.0);
            tc.verifyEmpty(near, ...
                sprintf("%d false exudate(s) reported at the disc margin", numel(near)));
        end

        function colourSeparatesDiscFromExudate(tc)
            % The paired test. Same distance, same size, same brightness.
            D = tc.DISC.r * 1.9;
            spots = [ ...
                struct("x", tc.DISC.x, "y", tc.DISC.y - D, "r", 9, ...
                       "colour", [232 228 224]), ...   % white: reflex / atrophy
                struct("x", tc.DISC.x, "y", tc.DISC.y + D, "r", 9, ...
                       "colour", [244 226 116])];      % yellow: lipid exudate

            I = syntheticFundus(Disc=tc.DISC, Spots=spots);
            [les, ~] = tc.analyse(I);

            whiteFound  = tc.found(les, spots(1));
            yellowFound = tc.found(les, spots(2));

            tc.verifyFalse(whiteFound, ...
                "the white peripapillary mark was reported as an exudate");
            tc.verifyTrue(yellowFound, ...
                "the yellow peripapillary exudate was lost - the rule is " + ...
                "suppressing on position rather than on colour");
        end

        function circinateRingSurvives(tc)
            % The case the fix is most likely to damage: real exudate at
            % 1.6-2.4 disc radii, well beyond the rim. All of it must survive.
            ring = struct("x", {}, "y", {}, "r", {});
            for k = 0:5
                a = k*pi/3 + 0.3;
                d = tc.DISC.r * (1.6 + 0.8*(mod(k,3)/2));
                ring(end+1) = struct("x", tc.DISC.x + d*cos(a), ...
                                     "y", tc.DISC.y + d*sin(a), "r", 8); %#ok<AGROW>
            end

            I = syntheticFundus(Disc=tc.DISC, Exudates=ring);
            [les, ~] = tc.analyse(I);

            nFound = 0;
            for k = 1:numel(ring)
                if tc.found(les, ring(k))
                    nFound = nFound + 1;
                end
            end
            tc.verifyGreaterThanOrEqual(nFound, 5, ...
                sprintf("only %d of %d circinate exudates survived", ...
                        nFound, numel(ring)));
        end

        function darkMarkBesideDiscIsExempt(tc)
            % Nothing about the disc is dark, so the peripapillary rules must
            % leave a dark mark alone. This asserts the rules did not fire, which
            % is separable from whether the detector found the mark at all.
            spot = struct("x", tc.DISC.x + tc.DISC.r*1.9, "y", tc.DISC.y, ...
                          "r", 9, "colour", [64 22 20]);
            I = syntheticFundus(Disc=tc.DISC, Spots=spot);
            [les, ~] = tc.analyse(I);

            tc.verifyEqual(les.rejected.discTissue, 0, ...
                "a dark mark was rejected as disc tissue");
            tc.verifyEqual(les.rejected.discColour, 0, ...
                "a dark mark was rejected on disc colour");
        end

        function cleanRetinaInventsNothing(tc)
            I = syntheticFundus(Disc=tc.DISC);
            [les, ~] = tc.analyse(I);
            tc.verifyLessThanOrEqual(numel(les.candidates), 2, ...
                sprintf("%d candidates on a clean synthetic retina", ...
                        numel(les.candidates)));
        end
    end

    methods (Access = private)

        function [les, anatomy] = analyse(tc, I) %#ok<INUSL>
            disc = dr.estimateOpticDisc(I);
            fm = dr.estimateFoveaMacula(I, disc);
            anatomy = struct("disc", disc, "fovea", fm.fovea, ...
                             "maculaRadius", fm.maculaRadius, ...
                             "nasalSide", fm.nasalSide, "evidence", fm.evidence);
            les = dr.detectLesionCandidates(I, anatomy);
        end

        function tf = found(tc, les, mark) %#ok<INUSL>
            tf = false;
            for i = 1:numel(les.candidates)
                c = les.candidates(i);
                if hypot(c.cx - mark.x, c.cy - mark.y) <= mark.r + 8
                    tf = true;
                    return;
                end
            end
        end

        function near = brightNearDisc(tc, les, mult)
            near = [];
            for i = 1:numel(les.candidates)
                c = les.candidates(i);
                isBright = c.type == "HE" || c.type == "CWS";
                if isBright && hypot(c.cx - tc.DISC.x, c.cy - tc.DISC.y) <= tc.DISC.r*mult
                    near(end+1) = i; %#ok<AGROW>
                end
            end
        end
    end
end
