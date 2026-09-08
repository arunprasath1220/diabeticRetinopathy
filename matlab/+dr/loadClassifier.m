function net = loadClassifier(matFile)
%LOADCLASSIFIER  Load the trained binary DR classifier.
%
%   NET = dr.loadClassifier() loads models/drClassifier.mat if it exists.
%   NET = dr.loadClassifier(FILE) loads a specific file.
%
%   If no trained network is present this returns [] rather than substituting an
%   untrained backbone. That distinction is carried all the way to the report: a
%   pipeline running without a classifier says so and shows no grade and no
%   Grad-CAM, because a MobileNet with a randomly initialised head will happily
%   emit a confident-looking probability that means nothing at all. The lesion
%   detector is classical and does not need the network, so it still runs - but
%   with nothing cross-checking it, which DR.ASSESSSEVERITY records as a doubt.
%
%   See also DR.TRAINCLASSIFIER.

    if nargin < 1 || isempty(matFile)
        here = fileparts(fileparts(mfilename("fullpath")));   % matlab/
        matFile = fullfile(here, "models", "drClassifier.mat");
    end

    net = [];
    if ~isfile(matFile)
        warning("dr:loadClassifier:missing", ...
            ['No trained classifier at %s. Steps 2 and 3 will be skipped and ' ...
             'reported as unavailable; run dr.trainClassifier to produce one.'], ...
            matFile);
        return;
    end

    S = load(matFile);
    if isfield(S, "net")
        net = S.net;
    elseif isfield(S, "trainedNet")
        net = S.trainedNet;
    else
        f = fieldnames(S);
        net = S.(f{1});
    end
end
