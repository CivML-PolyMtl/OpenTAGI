%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         infoGAN_V1
% Description:  Generative Adversarial Networks to Mnist
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      January 24, 2021
% Updated:      August 19, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha nguyen & James-A. Goulet. All rights reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
seed = 1123;
rng(seed)
%% Data
modelName = 'infoGAN_V1';
dataName  = 'mnist';
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
x            = single(x)/255;
x            = (x-1.3093e-01);%/std(reshape(x, [numel(x), 1]));
imgSize      = [28 28 1];
x            = dp.cvr2img(x, imgSize);
y            = single(y(idxN));
yref         = y;
nx           = prod(imgSize); 
x            = x(:,:,:, idxN);
netD.seed     = seed;

%% Discriminator
netD.task           = 'sharing';
netD.modelName      = modelName;
netD.dataName       = dataName;
netD.cd             = cd;
netD.maxEpoch       = 50; 
netD.savedEpoch     = 25;
% GPU 1: yes; 0: no
netD.gpu            = true;
% Data type object half or double precision
netD.dtype          = 'single';   
% Number of input covariates
netD.nx             = prod(imgSize); 
% Number of output responses
netD.ny             = 256;   
% Batch size 
netD.batchSize      = 16;  
netD.repBatchSize   = 1;
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling
netD.layer          = [2       2         6      4       2       6       4      1];
netD.filter         = [1       32        32     32      64      64      64     1];
netD.kernelSize     = [3       1         3      3       1       3       1      1];
netD.padding        = [1       0         1      1       0       1       0      0];
netD.stride         = [1       0         2      1       0       2       0      0];
netD.nodes          = [netD.nx 0         0      0       0       0       0      netD.ny]; 
netD.actFunIdx      = [0       6         0      0       6       0       0      6];
netD.paddingType    = [1       0         2      1       0       2       0      0];
netD.imgW           = [28      0         0      0       0       0       0      0]; 
netD.imgH           = [28      0         0      0       0       0       0      0];
% Observations standard deviation
netD.sv              = nan*ones(1,1);  
netD.initParamType   = 'He';
% Maximal number of learnign epoch
netD.convariateEstm  = true;
netD.lastLayerUpdate = false;
%% Discriminator
% GPU 1: yes; 0: no
netP.task           = 'discrimination';
netP.gpu            = netD.gpu;
% Data type object half or double precision
netP.dtype          = 'single';   
% Number of input covariates
netP.nx             = netD.ny; 
% Number of output responses
netP.ny             = 1;   
% Batch size 
netP.batchSize      = netD.batchSize; 
netP.repBatchSize   = netD.repBatchSize;
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling
netP.imgSize        = imgSize;
netP.layer          = [netD.layer(end)          1];
netP.filter         = [netD.filter(end)         1];
netP.kernelSize     = [netD.kernelSize(end)     1];
netP.padding        = [netD.padding(end)        0];
netP.stride         = [netD.stride(end)         0];
netP.nodes          = [netD.nodes(end)          netP.ny]; 
netP.actFunIdx      = [6                        0];
netP.paddingType    = [netD.paddingType(end)    0];
netP.imgW           = [netD.imgW(end)           0]; 
netP.imgH           = [netD.imgH(end)           0]; 
% Observations standard deviation
netP.sv             = 3*ones(1,1);  
netP.svDecayFactor  = 0.975;
netP.svmin          = 0.3;
netP.initParamType  = 'He';
% Maximal number of learnign epoch
netP.maxEpoch        = netD.maxEpoch;   
netP.convariateEstm  = true;
netP.lastLayerUpdate = true;
net.isUdidx          = false;
%% Q net 
% GPU 1: yes; 0: no
netQ.task           = 'classification';
netQ.gpu            = netD.gpu;
% Data type object half or double precision
netQ.dtype          = 'single';   
netQ.labels         = yref;
netQ.numClasses     = Nclass;
netQ.numContVar     = 2;
netQ.numCatVar      = 1;
% Number of input covariates
netQ.nx             = netD.ny; 
% Number of output responses
[yl, ~]             = dp.encoder(yref, netQ.numClasses, netQ.dtype);     
netQ.ny             = size(yl, 2) * netQ.numCatVar + netQ.numContVar;   
netQ.nye            = 4 * netQ.numCatVar + netQ.numContVar;
% Batch size 
netQ.batchSize      = netD.batchSize; 
netQ.repBatchSize   = netD.repBatchSize;
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling
netQ.imgSize        = imgSize;
netQ.layer          = [netD.layer(end)          1   1];
netQ.filter         = [netD.filter(end)         1   1];
netQ.kernelSize     = [netD.kernelSize(end)     1   1];
netQ.padding        = [netD.padding(end)        0   0];
netQ.stride         = [netD.stride(end)         0   0];
netQ.nodes          = [netD.nodes(end)          300 netQ.ny]; 
netQ.actFunIdx      = [6                        6   0];
netQ.paddingType    = [netD.paddingType(end)    0   0];
netQ.imgW           = [netD.imgW(end)           0   0]; 
netQ.imgH           = [netD.imgH(end)           0   0];
% Observations standard deviation
netQ.sv             = 3*ones(1,1); 
netQ.svDecayFactor  = 0.975;
netQ.svmin          = 0.3;
netQ.initParamType  = 'He';
% Maximal number of learnign epoch
netQ.maxEpoch        = netD.maxEpoch;   
netQ.convariateEstm  = true;
netQ.lastLayerUpdate = true;
netQ.isUdidx         = true;
%% Generator
% GPU 1: yes; 0: no
netG.task           = 'generation';
netG.gpu            = netD.gpu;
% Data type object half or double precision
netG.dtype          = 'single';   
netG.labels         = yref;
% Number of input covariates
netG.nx             = 62+netQ.ny; 
% Number of output responses
netG.ny             = prod(imgSize);   
% Batch size 
netG.batchSize      = netD.batchSize; 
netG.repBatchSize   = netD.repBatchSize;
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling
netG.imgSize        = imgSize;
netG.layer          = [1        1         21    21     21];
netG.filter         = [1        64        64    32     1];
netG.kernelSize     = [0        3         3     3      1];
netG.padding        = [0        1         1     1      0];
netG.stride         = [0        1         2     2      0];
netG.nodes          = [netG.nx  49*64     0     0      netG.ny]; 
netG.actFunIdx      = [0        4         4     4      0];
netG.paddingType    = [0        1         2     2      0];
netG.imgW           = [0        7         0     0      0]; 
netG.imgH           = [0        7         0     0      0]; 
netG.sx             = zeros(1, 1, netG.dtype);
% Observations standard deviation
netG.sv             = nan*ones(1,1); 
netG.initParamType  = 'He';
% Maximal number of learnign epoch
netG.maxEpoch       = netD.maxEpoch;   
netG.lastLayerUpdate = false;

%% Pretrained model directory
trainedModelDir = [];

%% Run
task.runInfoGAN(netD, netG, netQ, netP, x, trainIdx, trainedModelDir);

