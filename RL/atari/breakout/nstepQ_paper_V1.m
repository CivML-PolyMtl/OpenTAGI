%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         nstepQ_paper_V1
% Description:  nstep-Q learning for breakout
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      April 08, 2021
% Updated:      August 19, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha nguyen & James-A. Goulet. All rights reserved. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
% clc
close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE

seed = 0;
rng(seed)
%% Data
modelName        = 'nstepQ_V1';
dataName         = 'breakout';
env              = breakoutNoFrameskipEnv();
% RL hyperparameters 
rl.numEpisodes   = 50000;
rl.numStepsEps   = 18000;
rl.numFrameskip  = 4;
rl.numFrames     = 4;
rl.maxNumSteps   = round(40E6 / rl.numFrameskip);
rl.imgSize       = [84 84 4];
rl.gamma         = 0.99;
rl.maxMemory     = 1E1;
rl.actionSpace   = 1;
rl.numActions    = double(env.open_env.action_space.n);
rl.rewardSpace   = 1;
rl.stateSpace    = prod(rl.imgSize);
rl.targetUpdate  = 1; % Update the target network after N times
rl.initialMemory = 1E4;
rl.resizeMethod  = 'scale';
rl.rewardNormaliztion = false; 
rl.rewardScaling = false;
rl.noptepochs    = 1;
netP.rl          = rl;
netP.seed        = seed;

%% Policy net
netP.task           = 'classification';
netP.modelName      = modelName;
netP.dataName       = dataName;
netP.cd             = cd;
netP.savedUpdate    = 1E4;
netP.savedEpisode   = 200;
netP.rl.stepUpdate  = 128;
netP.logInterval    = 200;
% GPU 1: yes; 0: no
netP.gpu            = true;
netP.cuda           = true;
% Data type object half or double precision
netP.dtype          = 'single';
% Number of input covariates
netP.nx             = prod(netP.rl.imgSize); 
% Number of output responses
netP.ny             = netP.rl.numActions;   
% Batch size 
netP.batchSize      = 32; 
netP.repBatchSize   = 1;
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling
netP.imgSize        = netP.rl.imgSize;
netP.layer          = [2       2     2     1    1];
netP.filter         = [4       16    32    1    1];
netP.kernelSize     = [8       4     1     1    1];
netP.padding        = [0       0     0     0    0];
netP.stride         = [4       2     0     0    0];
netP.nodes          = [netP.nx 0     0     256  netP.ny];  
netP.actFunIdx      = [0       4     4     4    0];
netP.paddingType    = [0       0     0     0    0 ];
netP.actBound       = [1       1     1     1    1];
% Observations standard deviation
netP.sv             = 2*ones(1,1, netP.dtype);   
netP.svmin          = 0.3;
netP.svDecayFactor  = 0.9999;
% Parameter initialization | {Xavier, He}
netP.initParamType  = 'He';
% Update type
netP.lastLayerUpdate = true; 
netP.isUdidx         = true;% Only needed for CUDA compiler

%% Load model from results's directory 
trainedModelDir = []; 
startEpisode = 0;

%% Run
task.nstepQwithImagesCUDA(netP, env, trainedModelDir, startEpisode);
