function results = runAllTests()
%RUNALLTESTS  Run the MATLAB test suite for the DR pipeline.
%
%   results = runAllTests()
%
%   Adds the package parent to the path first, so the suite runs from a clean
%   session without the caller having to set anything up.

    here = fileparts(mfilename("fullpath"));
    root = fileparts(here);
    addpath(root);                    % so +dr resolves
    addpath(here);                    % so syntheticFundus resolves
    addpath(fullfile(root, "simulink"));

    suite = matlab.unittest.TestSuite.fromFolder(here);
    runner = matlab.unittest.TestRunner.withTextOutput( ...
        "OutputDetail", matlab.unittest.Verbosity.Detailed);

    results = runner.run(suite);

    fprintf("\n%d passed, %d failed, %d incomplete\n", ...
        nnz([results.Passed]), nnz([results.Failed]), nnz([results.Incomplete]));
end
