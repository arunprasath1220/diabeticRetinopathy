function [net, info] = trainClassifier(dataRoot, opts)
%TRAINCLASSIFIER  Transfer-learn a binary DR classifier on a labelled fundus set.
%
%   [NET, INFO] = dr.trainClassifier(DATAROOT) expects DATAROOT to hold one
%   subfolder per class:
%
%       DATAROOT/no_dr/*.jpg
%       DATAROOT/dr/*.jpg
%
%   Options (name-value):
%     Backbone      "mobilenetv2" (default) | "resnet18" | "efficientnetb0"
%     Split         [train val test] fractions, default [0.7 0.15 0.15]
%     MaxEpochs     default 12
%     MiniBatchSize default 32
%     OutputFile    where to save, default matlab/models/drClassifier.mat
%
%   The head is replaced and the backbone fine-tuned at a reduced learning rate.
%   A fundus photograph shares low-level structure with natural images - edges,
%   blobs, texture - but nothing at all above that, so freezing the backbone
%   entirely underfits and training from scratch on a screening-sized dataset
%   overfits. Fine-tuning with a small base rate and a large head rate is the
%   middle path, and the 10x head multiplier below is what implements it.
%
%   ---- On class balance ----
%   Screening data is heavily skewed toward "no DR", and a network trained on it
%   without correction reaches high accuracy by answering "no DR" always. The
%   sampling below equalises the classes for training. It does NOT reweight the
%   validation set, because validation has to reflect the population the system
%   will actually meet, and a balanced validation figure would flatter the model
%   in exactly the direction that matters least clinically.

    arguments
        dataRoot (1,1) string
        opts.Backbone (1,1) string = "mobilenetv2"
        opts.Split (1,3) double = [0.7 0.15 0.15]
        opts.MaxEpochs (1,1) double = 12
        opts.MiniBatchSize (1,1) double = 32
        opts.OutputFile (1,1) string = ""
    end

    C = dr.Config();
    inputSize = [C.model.inputSize 3];

    if opts.OutputFile == ""
        here = fileparts(fileparts(mfilename("fullpath")));
        opts.OutputFile = fullfile(here, "models", "drClassifier.mat");
    end

    imds = imageDatastore(dataRoot, ...
        "IncludeSubfolders", true, "LabelSource", "foldernames");

    [trainDs, valDs, testDs] = splitEachLabel(imds, ...
        opts.Split(1), opts.Split(2), opts.Split(3), "randomized");

    % Equalise the training classes only.
    counts = countEachLabel(trainDs);
    nPer = min(counts.Count);
    trainDs = splitEachLabel(trainDs, nPer, "randomized");

    % Augmentation the camera actually justifies: a fundus can arrive at any
    % rotation and either handedness, and rural capture varies in scale and
    % exposure. Nothing here changes colour balance, because the downstream
    % lesion colour test depends on it.
    aug = imageDataAugmenter( ...
        "RandRotation", [-180 180], ...
        "RandXReflection", true, ...
        "RandScale", [0.9 1.1], ...
        "RandXTranslation", [-10 10], ...
        "RandYTranslation", [-10 10]);

    trainAug = augmentedImageDatastore(inputSize, trainDs, "DataAugmentation", aug);
    valAug   = augmentedImageDatastore(inputSize, valDs);
    testAug  = augmentedImageDatastore(inputSize, testDs);

    numClasses = numel(categories(imds.Labels));

    % imagePretrainedNetwork replaces the head for the requested class count in
    % one call on recent releases; the manual path covers older ones.
    if exist("imagePretrainedNetwork", "file") == 2
        lgraph = imagePretrainedNetwork(opts.Backbone, NumClasses=numClasses);
    else
        lgraph = buildLegacyHead(opts.Backbone, numClasses);
    end

    options = trainingOptions("adam", ...
        "InitialLearnRate", 1e-4, ...
        "MaxEpochs", opts.MaxEpochs, ...
        "MiniBatchSize", opts.MiniBatchSize, ...
        "ValidationData", valAug, ...
        "ValidationFrequency", 30, ...
        "ValidationPatience", 5, ...
        "Shuffle", "every-epoch", ...
        "OutputNetwork", "best-validation-loss", ...
        "Plots", "training-progress", ...
        "Verbose", true);

    [net, info] = trainnet(trainAug, lgraph, "crossentropy", options);

    % Held-out evaluation, reported with the metrics the screening case needs
    % rather than accuracy alone.
    scores = minibatchpredict(net, testAug);
    trueLabels = testDs.Labels;
    metrics = dr.evaluateModel(scores, trueLabels);
    info.testMetrics = metrics;

    outDir = fileparts(opts.OutputFile);
    if ~isfolder(outDir)
        mkdir(outDir);
    end
    classNames = categories(imds.Labels); %#ok<NASGU>
    save(opts.OutputFile, "net", "classNames", "metrics");
    fprintf("Saved classifier to %s\n", opts.OutputFile);
end

function lgraph = buildLegacyHead(backbone, numClasses)
%BUILDLEGACYHEAD  Replace the classification head on an older release.
    switch backbone
        case "mobilenetv2", base = mobilenetv2;
        case "resnet18",    base = resnet18;
        otherwise
            error("dr:trainClassifier:backbone", ...
                  "unsupported backbone %s on this release", backbone);
    end
    lgraph = layerGraph(base);
    learnable = lgraph.Layers(end-2).Name;
    outputLayer = lgraph.Layers(end).Name;

    newFc = fullyConnectedLayer(numClasses, ...
        "Name", "dr_fc", ...
        "WeightLearnRateFactor", 10, ...     % head learns fast, backbone slow
        "BiasLearnRateFactor", 10);
    lgraph = replaceLayer(lgraph, learnable, newFc);
    lgraph = replaceLayer(lgraph, outputLayer, softmaxLayer("Name","dr_softmax"));
end
