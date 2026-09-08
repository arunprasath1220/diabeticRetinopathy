function result = classifyImage(net, I)
%CLASSIFYIMAGE  Step 2: run the binary DR classifier.
%
%   RESULT = dr.classifyImage(NET, I) returns probs, predClass (1-based) and
%   elapsedMs.
%
%   The preprocessing has to match what the network was trained on exactly, and
%   the two places it is written must not drift apart - so both this and
%   DR.COMPUTEGRADCAM call PREPAREINPUT below rather than each rolling their own.
%   A Grad-CAM computed on differently scaled input explains a different image
%   than the one that was graded, which is a subtle way for an explanation to be
%   confidently wrong.

    t0 = tic;
    x = dr.prepareInput(I);
    scores = predict(net, x);
    probs = double(scores(:))';
    [~, predClass] = max(probs);

    result = struct("probs", probs, "predClass", predClass, ...
                    "elapsedMs", toc(t0)*1000);
end
