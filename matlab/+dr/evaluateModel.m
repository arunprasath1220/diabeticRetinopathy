function m = evaluateModel(scores, trueLabels, positiveClass)
%EVALUATEMODEL  The metrics a screening system is actually judged on.
%
%   M = dr.evaluateModel(SCORES, TRUELABELS) where SCORES is N-by-2 and
%   TRUELABELS is a categorical vector. Returns accuracy, sensitivity,
%   specificity, precision, recall, f1, auc, the confusion matrix, the ROC, the
%   operating point chosen for high sensitivity, and a calibration assessment.
%
%   ---- Why accuracy is reported but not used ----
%   Screening data is heavily skewed toward "no DR". A model that answers "no DR"
%   always scores well on accuracy and is worthless. Sensitivity at a fixed
%   specificity is the figure that describes whether the system finds disease,
%   and the threshold is chosen on that basis rather than left at 0.5 - see
%   TARGETSENSITIVITY below. Missing a referable eye costs sight; a false alarm
%   costs one specialist appointment, and the two are not comparable.
%
%   ---- Calibration ----
%   A probability that is used to triage has to mean what it says. Expected
%   calibration error and the Brier score measure that directly, and both are
%   reported because a model can be well ordered (high AUC) and still badly
%   calibrated, which is exactly the failure that makes a "confidence" figure
%   misleading on a report.

    arguments
        scores double
        trueLabels categorical
        positiveClass (1,1) double = 2
    end

    TARGETSENSITIVITY = 0.90;

    classes = categories(trueLabels);
    posName = classes{positiveClass};
    yTrue = double(trueLabels == posName);
    p = scores(:, positiveClass);

    % ---- ROC and AUC ----------------------------------------------------
    [rocX, rocY, rocT, auc] = perfcurve(yTrue, p, 1);
    m.auc = auc;
    m.roc = struct("fpr", rocX, "tpr", rocY, "thresholds", rocT);

    % ---- Operating point for high-sensitivity screening ------------------
    % The lowest false-positive rate that still reaches the target sensitivity.
    ok = find(rocY >= TARGETSENSITIVITY, 1, "first");
    if isempty(ok)
        ok = numel(rocY);
    end
    m.operatingThreshold = rocT(ok);
    m.targetSensitivity = TARGETSENSITIVITY;

    yHat = double(p >= m.operatingThreshold);

    % ---- Confusion matrix and everything read off it ---------------------
    Cm = confusionmat(yTrue, yHat, "Order", [0 1]);
    tn = Cm(1,1);  fp = Cm(1,2);  fn = Cm(2,1);  tp = Cm(2,2);
    m.confusion = Cm;

    m.accuracy    = (tp + tn) / max(1, tp+tn+fp+fn);
    m.sensitivity = tp / max(1, tp+fn);
    m.specificity = tn / max(1, tn+fp);
    m.precision   = tp / max(1, tp+fp);
    m.recall      = m.sensitivity;
    m.f1 = 2*m.precision*m.recall / max(eps, m.precision + m.recall);

    % ---- Calibration -----------------------------------------------------
    m.brier = mean((p - yTrue).^2);

    nBins = 10;
    edges = linspace(0, 1, nBins+1);
    bin = discretize(p, edges);
    ece = 0;
    reliability = nan(nBins, 3);          % [meanPredicted observedRate count]
    for b = 1:nBins
        sel = (bin == b);
        cnt = nnz(sel);
        if cnt == 0
            continue;
        end
        meanPred = mean(p(sel));
        observed = mean(yTrue(sel));
        reliability(b,:) = [meanPred observed cnt];
        ece = ece + (cnt/numel(p)) * abs(meanPred - observed);
    end
    m.ece = ece;
    m.reliability = reliability;

    % ---- Uncertainty ------------------------------------------------------
    % Predictive entropy, normalised to 0..1. Near 1 the model is at its decision
    % boundary, which on a screening report should read as "unresolved", not as a
    % 50% probability of disease.
    q = min(max(scores, eps), 1-eps);
    m.entropy = -sum(q .* log(q), 2) / log(size(scores,2));
end
