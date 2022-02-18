%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         adversarialAttack_V2
% Description:  Apply 3 conv to cinic 10 with 2 separate netAs
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      May 21, 2021
% Updated:      May 21, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 
% clc
close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
rng(1223)
% gpuDevice(1);
%% Data
Nsamples  = 20;
netA.testing = true;
modelName = '3cpnv_V1';
dataName  = 'cifar10';
ltrain    = load('data/cifar10train.mat');
trainImdb = ltrain.trainImdb;
ltest     = load('data/cifar10test.mat');
testImdb  = ltest.testImdb;
testImdb.img         = testImdb.img(:, :, :, 1 : Nsamples);
testImdb.digitlabels = testImdb.digitlabels(1 : Nsamples);
testImdb.classObs    = trainImdb.classObs;
testImdb.classIdx    = trainImdb.classIdx;
testImdb.numImages   = Nsamples;
imgSize   = [32 32 3];
% imgStat   = [0.485, 0.456, 0.406; % mean
%              0.229, 0.224, 0.225];% std
imgStat   = [0.485, 0.456, 0.406; % mean
             1, 1, 1];% std   
%% Neural Network properties
netA.task           = 'classification';
netA.modelName      = modelName;
netA.dataName       = dataName;
netA.cd             = cd;
% Save netA after N epoch
netA.maxEpoch       = 75; 
netA.savedEpoch     = 10;
% GPU 1: yes; 0: no
netA.gpu            = true;
netA.cuda           = true;
netA.numDevices     = 1;
% Data type object half or double precision
netA.dtype          = 'single';
% Number of input covariates
netA.imgSize        = imgSize;
netA.resizeMode     = false;
netA.resizedSize    = imgSize;
netA.nx             = prod(netA.imgSize); 
netA.imgStat        = imgStat;
% Number of output responses
ny                 = trainImdb.numOutputs; 
netA.ny             = ny;   
netA.nl             = ny;
netA.nv2            = ny; 
netA.nye            = trainImdb.numEnOutputs; % Num. of hidden states
% Number of classes
netA.numClasses     = trainImdb.numClasses;
% Batch size 
netA.batchSize      = 10; 
netA.repBatchSize   = 1;
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling
netA.imgSize        = imgSize;
netA.layer          = [2       2   6   3       2   6   4       2   6   4    1    1];
netA.filter         = [3       32  32  32      32  32  32      64  64  64   1    1];
netA.kernelSize     = [5       1   3   5       1   3   5       1   3   1    1    1];
netA.padding        = [2       0   1   2       0   1   2       0   2   0    0    0];
netA.stride         = [1       0   2   1       0   2   1       0   1   0    0    0];
netA.nodes          = [netA.nx 0   0   0       0   0   0       0   0   0    64   netA.ny];  
netA.actFunIdx      = [0       4   0   0       4   0   0       4   0   0    4    0];
netA.paddingType    = [1       0   2   1       0   2   1       0   2   0    0    0];
netA.actBound       = ones(size(netA.layer));
% Observations standard deviation
netA.learnSv        = 0;
netA.sv             = 1E-6*ones(1,1, netA.dtype);      
netA.svDecayFactor  = 0.975;
netA.svmin          = 0.2;
netA.Sx             = cast(0.05.^2, netA.dtype);
% Runing average
netA.normMomentum   = 0.9;

% Parameter initialization
netA.initParamType  = 'He';
netA.gainSw         = ones(1, length(netA.layer) - 1);
netA.gainMw         = ones(1, length(netA.layer) - 1);
netA.gainSb         = ones(1, length(netA.layer) - 1);
netA.gainMb         = ones(1, length(netA.layer) - 1);

% Last layer update
netA.isUdidx         = true;
netA.wxupdate        = true;
netA.collectDev      = false;
netA.convariateEstm  = true;
netA.errorRateEval   = true;
%% Neural Network properties
netD.task           = 'classification';
netD.modelName      = modelName;
netD.dataName       = dataName;
netD.cd             = cd;
% Save netD after N epoch
netD.maxEpoch       = 75; 
netD.savedEpoch     = 10;
% GPU 1: yes; 0: no
netD.gpu            = true;
netD.cuda           = true;
netD.numDevices     = 1;
% Data type object half or double precision
netD.dtype          = 'single';
% Number of input covariates
netD.imgSize        = imgSize;
netD.resizeMode     = false;
netD.resizedSize    = imgSize;
netD.nx             = prod(netD.imgSize); 
netD.imgStat        = imgStat;
% Number of output responses
ny                 = trainImdb.numOutputs; 
netD.ny             = ny;   
netD.nl             = ny;
netD.nv2            = ny; 
netD.nye            = trainImdb.numEnOutputs; % Num. of hidden states
% Number of classes
netD.numClasses     = trainImdb.numClasses;
% Batch size 
netD.batchSize      = 10; 
netD.repBatchSize   = 1;
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling
netD.imgSize        = imgSize;
netD.layer          = [2       2   5   3       2   5   4       2   5   4    1    1];
netD.filter         = [3       32  32  32      32  32  32      64  64  64   1    1];
netD.kernelSize     = [5       1   3   5       1   3   5       1   3   1    1    1];
netD.padding        = [2       0   1   2       0   1   2       0   2   0    0    0];
netD.stride         = [1       0   2   1       0   2   1       0   1   0    0    0];
netD.nodes          = [netD.nx 0   0   0       0   0   0       0   0   0    64   netD.ny];  
netD.actFunIdx      = [0       4   0   0       4   0   0       4   0   0    4    0];
netD.paddingType    = [1       0   2   1       0   2   1       0   2   0    0    0];
netD.actBound       = ones(size(netD.layer));
% Observations standard deviation
netD.learnSv        = 0;
netD.sv             = 1E-6*ones(1,1, netD.dtype);      
netD.svDecayFactor  = 0.975;
netD.svmin          = 0.2;
netD.Sx             = cast(0.05.^2, netD.dtype);
% Runing average
netD.normMomentum   = 0.9;

% Parameter initialization
netD.initParamType  = 'He';
netD.gainSw         = ones(1, length(netD.layer) - 1);
netD.gainMw         = ones(1, length(netD.layer) - 1);
netD.gainSb         = ones(1, length(netD.layer) - 1);
netD.gainMb         = ones(1, length(netD.layer) - 1);

% Last layer update
netD.isUdidx         = true;
netD.wxupdate        = true;
netD.collectDev      = false;
netD.convariateEstm  = true;
netD.errorRateEval   = true;
%% Pretrained model directory
trainedModelDirA = ['results/dropconetectE50_BN_V1_E50_cifar10']; 
trainedModelDirD = ['results/dropconetectE50_LN_V1_E50_cifar10']; 
initModelDir    = [];
initEpoch = 0;

%% Run
task.adversarialAttack_V2(netA, netD, testImdb, trainedModelDirA, trainedModelDirD, initEpoch)


