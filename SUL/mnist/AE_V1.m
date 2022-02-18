%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         AE_V1
% Description:  Apply autoencoder to Mnist
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      January 13, 2021
% Updated:      August 19, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha nguyen & James-A. Goulet. All rights reserved. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
% clc
% close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
seed = 1123;
rng(seed);
%% Data
modelName    = 'AE_V1';
dataName     = 'mnist';
path         = char([cd ,'/data/']);
load(char([path, '/mnistTrain_V2.mat']))
x            = x_obs;
y            = y_obs;
path         = char([cd ,'/data/']);
load(char([path, '/mnistTest_V2.mat']))
x            = [x; x_obs];
y            = [y; y_obs];
Nclass       = 10;
idxN         = y < Nclass;
trainIdx     = 1:60000;
testIdx      = (length(trainIdx)+1):size(x,1);
x            = single(x);
x            = (x-mean(reshape(x, [numel(x), 1])))/std(reshape(x, [numel(x), 1]));
imgSize      = [28 28 1];
x            = dp.cvr2img(x, imgSize);
y            = single(y(idxN));
yref         = y;
nx           = prod(imgSize); 
x            = x(:,:,:, idxN);
netE.seed     = seed;

%% Encoding net
netE.task           = 'encoding';
netE.modelName      = modelName;
netE.dataName       = dataName;
netE.cd             = cd;
netE.savedEpoch     = 5;
netE.maxEpoch       = 10; 
% GPU 1: yes; 0: no
netE.gpu            = true;
% Data type object half or double precision
netE.dtype          = 'single';   
% Number of input covariates
netE.nx             = nx; 
% Number of output responses
netE.ny             = 4; 
netE.nl             = netE.ny; 
netE.nv2            = netE.ny;
% Batch size 
netE.batchSize      = 10;  
netE.repBatchSize   = 1;
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling
netE.imgSize        = imgSize;
netE.imgSize        = imgSize;
netE.layer          = [2       2         6      4       2       6       4      1    1];
netE.filter         = [1       32        32     32      64      64      64     1    1];
netE.kernelSize     = [3       1         3      3       1       3       1      1    1];
netE.padding        = [1       0         1      1       0       1       0      0    0];
netE.stride         = [1       0         2      1       0       2       0      0    0];
netE.nodes          = [netE.nx 0         0      0       0       0       0      128  netE.ny]; 
netE.actFunIdx      = [0       4         0      0       4       0       0      4    0];
netE.paddingType    = [1       0         2      1       0       2       0      0    0];
netE.imgW           = [28      0         0      0       0       0       0      0    0]; 
netE.imgH           = [28      0         0      0       0       0       0      0    0]; 
% Observations standard deviation
netE.sv             = nan*ones(1, 1, netE.dtype); 
netE.initParamType  = 'He';
% Update layer
netE.convariateEstm  = false;
netE.lastLayerUpdate = false;
%% Decoding net
% GPU 1: yes; 0: no
netD.task           = 'decoding';
netD.gpu            = netE.gpu;
% Data type object half or double precision
netD.dtype          = 'single';   
% Number of input covariates
netD.nx             = netE.ny; 
% Number of output responses
netD.ny             = netE.nx;   
% Batch size 
netD.batchSize      = netE.batchSize; 
netD.repBatchSize   = netE.repBatchSize;
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling
netD.imgSize        = imgSize;
netD.layer          = [1        1       21      21      21];
netD.filter         = [1        64      64      32      1];
netD.kernelSize     = [1        3       3       3       1];
netD.padding        = [0        1       1       1       0];
netD.stride         = [0        2       2       1       0];
netD.nodes          = [netD.nx  49*64   0       0       netD.ny]; 
netD.actFunIdx      = [0        4       4       4       0];
netD.paddingType    = [0        2       2       1       0];
netD.imgW           = [28       7       0       0       0]; 
netD.imgH           = [28       7       0       0       0]; 
% Observations standard deviation
netD.sv             = 10 * ones(1, 1); 
netD.svDecayFactor  = 0.95;
netD.svmin          = 2;
netD.initParamType  = 'He';
% Infer hidden states of the first layer
netD.convariateEstm  = true;
netD.lastLayerUpdate = true;
netD.isUdidx         = false;

%% Pretrained model directory
trainedModelDir = [];

%% Run
task.runAE(netE, netD, x, trainIdx, trainedModelDir);


