%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         example_1D
% Description:  test full-covariance feed forward
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      July 20, 2021
% Updated:      August 19, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha Nguyen & James-A. Goulet. All rights reserved. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
% clc
%  close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
%% Data
rng(0)
modelName  = 'example_min';
dataName   = '1D';
ntrain     = 50;
ntest      = 20;
normalizeData = true;
a           = -1;
b           = 1;
% 
f          = @(x) x.^3;
x          = linspace(-5, 5, ntrain + ntest)';%[rand(1,ntrain)*2 - 1].*5;%linspace(-10, 10, ntrain);%-1:0.1:1;
idx        = randperm(ntrain + ntest, ntrain + ntest);
idxtrain   = idx(1 : ntrain);
idxtest    = idx(ntrain + 1 : end);
xtrain     = x(idxtrain, :);
xtest      = x(idxtest, :);
ytrainTrue = f(xtrain);
ytrain     = f(xtrain);
ytestTrue  = f(xtest);
ytest      = ytestTrue;
nx         = size(xtrain, 2);
ny         = size(ytrain, 2);
% [~, idx] = sort(xtrain);
% % plot(xtrain(idx), ytrain(idx));
maxXtrain = max(xtrain, [], 1);
minXtrain = min(xtrain, [], 1);
maxYtrain = max(ytrain);
minYtrain = min(ytrain);
if normalizeData    
    x      = normalize([xtrain; xtest],'range', [-1 1]);
    y      = normalize([ytrain; ytest],'range', [-1 1]);
    ytrain = y(1 : length(ytrain), :);
    ytest  = y(length(ytrain) + 1 : end);
    xtrain = x(1 : length(xtrain), :);
    xtest  = x(length(xtrain) + 1 : end, :);  
end
%% Add noise to observation
sv     = 0.01;
ytrain = ytrain + normrnd(0, sv, [ntrain, 1]);
ytest  = ytest + normrnd(0, sv, [ntest, 1]);
% [~, idx] = sort(xtrain);
% plot(xtrain(idx), ytrain(idx));

%% Initilize uncertainties for inputs
Sxtrain = ones(ntrain, 1) * (2/10).^2;
Sxtest  = ones(ntest, 1) * (2/10).^2;

%% Net
% GPU 1: yes; 0: no
net.task           = 'regression';
net.modelName      = modelName;
net.dataName       = dataName;
net.cd             = cd;
net.saveModel      = 1;
net.maxEpoch       = 100;  
% GPU
net.gpu            = false;
net.cuda           = false;% Run with cuda compiler version
net.numDevices     = 1;
% Data type object half or double precision
net.dtype          = 'single';
% Number of input covariates
net.nx             = nx; 
% Number of output responses
net.nl             = ny;
net.nv2            = ny;
net.ny             = 1*ny; 
% Batch size 
net.batchSize      = 10; 
net.repBatchSize   = 1; 
% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.layer          = [1         1   1   1];
net.nodes          = [net.nx    50  50  net.ny]; 
net.actFunIdx      = [0         4   4   0];
net.actBound       = [1         1   1   1];
% Observations standard deviation
net.learnSv        = 0;% Online noise learning
net.sv             = 1 * ones(1, 1);  
net.noiseType      = 'none';
% Parameter initialization
net.initParamType  = 'He';
if net.gpu
    xtrain  = gpuArray(single(xtrain));
    Sxtrain = gpuArray(single(Sxtrain));
    ytrain  = gpuArray(single(ytrain));
    xtest   = gpuArray(single(xtest));
    Sxtest  = gpuArray(single(Sxtest));
    ytest   = gpuArray(single(ytest));    
end
%% Run
task.runRegressionFullCov(net, xtrain, Sxtrain, ytrain, xtest, Sxtest, ytest)



