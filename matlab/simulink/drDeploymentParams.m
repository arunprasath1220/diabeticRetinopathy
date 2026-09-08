function P = drDeploymentParams()
%DRDEPLOYMENTPARAMS  Scenario parameters for the deployment simulation.
%
%   Every number here is a modelling assumption, not a measurement, and each one
%   is written down with where it came from so a reviewer can argue with it. A
%   deployment model whose assumptions are buried in block dialogs cannot be
%   challenged, and an unchallengeable capacity estimate is worth nothing.
%
%   See also BUILDDRSCREENINGMODEL.

% ---- Simulation horizon -------------------------------------------------
P.stepSeconds = 60;                 % one minute per step
P.simSeconds  = 30 * 24 * 3600;     % thirty days

% ---- Demand -------------------------------------------------------------
% The slide's target: district-scale, 100,000+ patients annually.
P.patientsPerDay = 100000 / 250;    % ~400/day over 250 working days
% Screening happens in camps, not as a steady stream. The Gaussian day-profile
% below concentrates arrivals around mid-morning, which is what produces the
% uplink surge the programme has to survive.
P.peakHour = 10.5;
P.dayWidthHours = 2.5;

% ---- Image capture ------------------------------------------------------
P.imageMB = 1.8;                    % JPEG from a portable fundus camera
% Ungradable rates in published rural screening programmes run 10-20% with
% non-mydriatic portable cameras; the high end is taken, because a capacity
% estimate that assumes the best case is the one that fails in the field.
P.ungradableRate = 0.18;
% Of those, most are recoverable if the operator is told immediately - which is
% the entire justification for putting the quality gate at the point of capture.
P.recaptureSuccessRate = 0.70;

% ---- Network ------------------------------------------------------------
% A modest district uplink. Sweep this: it is the parameter the model exists to
% answer questions about.
P.uplinkMbps = 8;

% ---- Compute ------------------------------------------------------------
P.inferencePerSecond = 12;          % images/s on one modest GPU worker
P.inferenceWorkers = 1;

% ---- Clinical review ----------------------------------------------------
% Referral rate across a screening population: prevalence of any DR is roughly
% a third in diagnosed diabetics, and this pipeline refers on any lesion or any
% doubt, so the operating referral rate is deliberately higher than prevalence.
P.referralRate = 0.35;
% Minutes of ophthalmologist time per flagged image, including the report.
P.reviewMinutes = 4;
end
