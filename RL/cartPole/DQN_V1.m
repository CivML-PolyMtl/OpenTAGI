%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         DQN_V1
% Description:  Deep Q learning for cart pole
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      November 05, 2020
% Updated:      August 19, 2020
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha nguyen & James-A. Goulet. All rights reserved. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 
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
modelName        = 'DQN_V1';
dataName         = 'cartPole';
env              = cartPoleEnv();
% RL hyperparameters 
rl.numEpisodes   = 10000;
rl.imgSize       = [1 1 1];
rl.gamma         = 0.99;
rl.maxMemory     = 50000;
rl.actionSpace   = 1;
rl.numActions    = double(env.open_env.action_space.n);
rl.rewardSpace   = 1;
rl.stateSpace    = prod(rl.imgSize);
rl.targetUpdate  = 1; % Update the target network after N times
rl.initialMemory = 1;
netP.rl          = rl;
netP.rl.maxSteps = 1E6;
netP.seed        = seed;

%% Sharing net
netP.task           = 'classification';
netP.modelName      = modelName;
netP.dataName       = dataName;
netP.cd             = cd;
netP.savedEpisode   = round(rl.numEpisodes/2);
% Data type object half or double precision
netP.dtype          = 'single';
% Number of input covariates
netP.nx             = 4; 
% Number of output responses
netP.ny             = netP.rl.numActions ;  
% Batch size 
netP.batchSize      = 10; 
netP.repBatchSize   = 1;
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling
netP.imgSize        = netP.rl.imgSize;
netP.layer          = [1        1   1];
netP.nodes          = [netP.nx  64  netP.ny];  
netP.actFunIdx      = [0        4   0];
netP.actBound       = [1        1   1];
% Observations standard deviation
netP.learnSv        = 0;
netP.learnSv        = 0;
netP.sv             = 2*ones(1,1, netP.dtype);   
netP.svmin          = 0.3;
netP.svDecayFactor   = 0.9999;
netP.lastLayerUpdate = true; 
% Parameter initialization | {Xavier, He}
netP.initParamType  = 'He';

%% Load model from results's directory
trainedModelDir = []; 
startEpisode = 0;
%% Run
task.DQN1net(netP, env, trainedModelDir, startEpisode);

