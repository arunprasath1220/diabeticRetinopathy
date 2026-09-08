function mdl = buildDRScreeningModel(mdlName)
%BUILDDRSCREENINGMODEL  Build the telemedicine deployment simulation.
%
%   MDL = buildDRScreeningModel() creates, saves and opens the model.
%
%   ---- What this model is for ----
%   The image pipeline answers "what is in this eye". It does not answer the
%   question a district health administrator actually has, which is whether one
%   screening programme can carry 100,000 patients a year on the staff, bandwidth
%   and specialist time available. That is a queueing question, and it is what
%   this model simulates:
%
%     arrivals -> capture -> quality gate -> uplink queue -> inference -> triage
%
%   ---- The three modelling choices that matter ----
%
%   * The quality gate sits BEFORE the upload, not after. Ungradable images are
%     the largest single load in rural screening, and rejecting them at the point
%     of capture both saves the bandwidth and lets the operator recapture while
%     the patient is still present. Modelling it the other way round understates
%     bandwidth by roughly the ungradable rate and puts the recapture rate at
%     nearly zero, because by then the patient has gone home.
%
%   * Bandwidth is a shared, finite link with a queue, because that is what a
%     district network is. A per-image average would hide the thing that actually
%     goes wrong, which is a morning camp of 200 patients saturating an uplink
%     sized for a daily mean.
%
%   * Referral load is the binding constraint, not compute. Inference is
%     milliseconds; an ophthalmologist reviewing a flagged image is minutes, and
%     there are few of them. Specialist-hours is therefore the headline output.
%
%   Parameters live in DRDEPLOYMENTPARAMS and are placed in the MODEL workspace,
%   not the base workspace, so the model carries its own scenario and a sweep
%   cannot be invalidated by whatever else a session happens to have loaded.
%
%   See also DRDEPLOYMENTPARAMS, DR.RUNPIPELINE.

    if nargin < 1 || isempty(mdlName)
        mdlName = "drScreeningDeployment";
    end
    mdl = char(mdlName);

    if bdIsLoaded(mdl)
        close_system(mdl, 0);
    end
    new_system(mdl);

    P = drDeploymentParams();

    % The scenario travels with the model. MATLAB Function blocks resolve `P` as
    % a parameter from here, which is why no block below needs EVALIN - and EVALIN
    % would not work anyway, since these blocks run under code generation
    % semantics even in normal simulation.
    ws = get_param(mdl, "ModelWorkspace");
    assignin(ws, "P", P);

    set_param(mdl, ...
        "SolverType", "Fixed-step", ...
        "Solver", "FixedStepDiscrete", ...
        "FixedStep", num2str(P.stepSeconds), ...
        "StartTime", "0", ...
        "StopTime", num2str(P.simSeconds));

    x = 40;  y = 60;  dx = 200;

    add_block("simulink/Sources/Clock", [mdl "/Clock"], "Position", pos(x, y));

    add_block("simulink/User-Defined Functions/MATLAB Function", ...
        [mdl "/Arrivals"], "Position", pos(x+dx, y));

    add_block("simulink/User-Defined Functions/MATLAB Function", ...
        [mdl "/QualityGate"], "Position", pos(x+2*dx, y));

    add_block("simulink/User-Defined Functions/MATLAB Function", ...
        [mdl "/UplinkQueue"], "Position", pos(x+3*dx, y));

    % A plain unit delay closes the queue loop. A Discrete-Time Integrator would
    % also work but multiplies by the sample time, so the accumulator would need
    % a compensating gain - one more place for a scenario change to go wrong
    % silently. The delay says exactly what it does: last step's backlog.
    add_block("simulink/Discrete/Unit Delay", [mdl "/Backlog"], ...
        "Position", pos(x+3*dx, y+110), ...
        "InitialCondition", "0", ...
        "SampleTime", num2str(P.stepSeconds));

    add_block("simulink/User-Defined Functions/MATLAB Function", ...
        [mdl "/Inference"], "Position", pos(x+4*dx, y));

    add_block("simulink/User-Defined Functions/MATLAB Function", ...
        [mdl "/Triage"], "Position", pos(x+5*dx, y));

    % Each body takes P as its last argument. Without the second call below,
    % Simulink would give P an input PORT and every block would need the
    % scenario wired to it; promoting it to a parameter is what lets the
    % diagram show only the flow of patients.
    setFcn([mdl "/Arrivals"],    arrivalsCode(),    "P");
    setFcn([mdl "/QualityGate"], qualityGateCode(), "P");
    setFcn([mdl "/UplinkQueue"], uplinkCode(),      "P");
    setFcn([mdl "/Inference"],   inferenceCode(),   "P");
    setFcn([mdl "/Triage"],      triageCode(),      "P");

    % ---- Logged outputs ---------------------------------------------------
    outs = ["Screened", "Ungradable", "UplinkBacklog", "Referred", "SpecialistHours"];
    for k = 1:numel(outs)
        add_block("simulink/Sinks/To Workspace", [mdl "/" char(outs(k))], ...
            "Position", pos(x+6*dx, y + (k-1)*70), ...
            "VariableName", char(outs(k)), ...
            "SampleTime", num2str(P.stepSeconds), ...
            "SaveFormat", "Timeseries");
    end

    add_block("simulink/Sinks/Scope", [mdl "/Monitor"], ...
        "Position", pos(x+6*dx, y+5*70), "NumInputPorts", "3");

    % ---- Wiring -------------------------------------------------------------
    add_line(mdl, "Clock/1",        "Arrivals/1",    "autorouting", "on");
    add_line(mdl, "Arrivals/1",     "QualityGate/1", "autorouting", "on");
    add_line(mdl, "QualityGate/1",  "UplinkQueue/1", "autorouting", "on");
    add_line(mdl, "Backlog/1",      "UplinkQueue/2", "autorouting", "on");
    add_line(mdl, "UplinkQueue/2",  "Backlog/1",     "autorouting", "on");
    add_line(mdl, "UplinkQueue/1",  "Inference/1",   "autorouting", "on");
    add_line(mdl, "Inference/1",    "Triage/1",      "autorouting", "on");

    add_line(mdl, "Arrivals/1",     "Screened/1",        "autorouting", "on");
    add_line(mdl, "QualityGate/2",  "Ungradable/1",      "autorouting", "on");
    add_line(mdl, "UplinkQueue/2",  "UplinkBacklog/1",   "autorouting", "on");
    add_line(mdl, "Triage/1",       "Referred/1",        "autorouting", "on");
    add_line(mdl, "Triage/2",       "SpecialistHours/1", "autorouting", "on");

    add_line(mdl, "Arrivals/1",    "Monitor/1", "autorouting", "on");
    add_line(mdl, "UplinkQueue/2", "Monitor/2", "autorouting", "on");
    add_line(mdl, "Triage/1",      "Monitor/3", "autorouting", "on");

    Simulink.BlockDiagram.arrangeSystem(mdl);
    save_system(mdl, fullfile(fileparts(mfilename("fullpath")), mdl));
    open_system(mdl);
end

% =========================================================================
function p = pos(x, y)
    p = [x y x+130 y+45];
end

function setFcn(blockPath, code, paramNames)
%SETFCN  Put MATLAB code into a MATLAB Function block.
%
%   Setting the Script creates chart data for every function argument, all of
%   it scoped as Input - which would put a port on the block. PARAMNAMES lists
%   the arguments that should instead resolve from the model workspace, so the
%   scenario is configuration rather than signal.
    chart = sfroot().find("-isa", "Stateflow.EMChart", "Path", blockPath);
    chart.Script = code;
    if nargin < 3
        return;
    end
    for k = 1:numel(paramNames)
        d = chart.find("-isa", "Stateflow.Data", "Name", char(paramNames(k)));
        if ~isempty(d)
            d(1).Scope = "Parameter";
        end
    end
end

% ---- Block bodies ------------------------------------------------------
% Written without POISSRND / BINORND on purpose. Those need code-generation
% support that a MATLAB Function block cannot assume, and the two algorithms
% below - Knuth for Poisson, a counted Bernoulli loop for the binomial - are
% short enough to read and depend on nothing but RAND.

function s = arrivalsCode()
s = strjoin([ ...
"function arrivals = fcn(t, P)"
"%#codegen"
"% Poisson arrivals shaped by a camp-day profile. Screening happens in camps,"
"% not as a steady stream, and a flat rate would never produce the morning"
"% surge that the uplink actually has to survive."
"hour = mod(t/3600, 24);"
"shape = exp(-0.5*((hour - P.peakHour)/P.dayWidthHours)^2);"
"lambda = P.patientsPerDay * shape * P.stepSeconds / ..."
"         (P.dayWidthHours*3600*sqrt(2*pi));"
"lambda = max(0, lambda);"
""
"% Knuth's method. The iteration cap keeps the loop bounded, which is what"
"% makes this safe to generate code from; at these arrival rates it is never"
"% approached."
"Lb = exp(-lambda);"
"k = 0;"
"pAcc = 1;"
"while pAcc > Lb && k < 1000"
"    k = k + 1;"
"    pAcc = pAcc * rand;"
"end"
"arrivals = double(max(0, k - 1));"
"end"], newline);
end

function s = qualityGateCode()
s = strjoin([ ...
"function [gradable, ungradable] = fcn(arrivals, P)"
"%#codegen"
"% The gate runs at the point of capture, before anything is uploaded."
"% Ungradable frames are the largest single load in rural screening, and"
"% catching them here saves the bandwidth AND lets the operator recapture while"
"% the patient is still in the room. Putting the gate after the upload"
"% understates bandwidth by the ungradable rate and drops the recapture rate to"
"% nearly zero, because by then the patient has gone home."
"failed = 0;"
"for i = 1:arrivals"
"    if rand < P.ungradableRate"
"        failed = failed + 1;"
"    end"
"end"
"recaptured = 0;"
"for i = 1:failed"
"    if rand < P.recaptureSuccessRate"
"        recaptured = recaptured + 1;"
"    end"
"end"
"gradable = arrivals - failed + recaptured;"
"ungradable = failed - recaptured;"
"end"], newline);
end

function s = uplinkCode()
s = strjoin([ ...
"function [uploaded, backlog] = fcn(gradable, prevBacklog, P)"
"%#codegen"
"% A shared, finite link with a queue - which is what a district network is."
"% A per-image average would hide the failure that actually happens: a morning"
"% camp saturating an uplink sized for a daily mean."
"capacity = P.uplinkMbps * P.stepSeconds / (P.imageMB * 8);"
"waiting = prevBacklog + gradable;"
"uploaded = min(waiting, capacity);"
"backlog = waiting - uploaded;"
"end"], newline);
end

function s = inferenceCode()
s = strjoin([ ...
"function graded = fcn(uploaded, P)"
"%#codegen"
"% Inference is not the bottleneck and the model should not pretend it is. One"
"% modest GPU worker clears thousands of images an hour. The cap exists only so"
"% that a scenario with no server at all can be swept."
"capacity = P.inferencePerSecond * P.stepSeconds * P.inferenceWorkers;"
"graded = min(uploaded, capacity);"
"end"], newline);
end

function s = triageCode()
s = strjoin([ ...
"function [referred, specialistHours] = fcn(graded, P)"
"%#codegen"
"% Referral load is what limits the programme, not compute. Inference is"
"% milliseconds; an ophthalmologist reviewing a flagged image is minutes, and"
"% there are few of them - so specialist-hours is the headline output."
"referred = graded * P.referralRate;"
"specialistHours = referred * P.reviewMinutes / 60;"
"end"], newline);
end
