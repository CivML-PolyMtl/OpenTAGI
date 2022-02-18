%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         resnet20Keras_V1
% Description:  Apply 2 residual network (Keras) layers to cifar 10
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      March 01, 2019
% Updated:      August 19, 2021
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
net.testing = false;
modelName = 'resnet20KerasE100_V5';
dataName  = 'cifar10';
ltrain    = load('data/cifar10train.mat');
trainImdb = ltrain.trainImdb;
trainImdb.classObs = single(trainImdb.classObs);
trainImdb.classIdx = single(trainImdb.classIdx);
ltest     = load('data/cifar10test.mat');
testImdb  = ltest.testImdb;
imgSize   = [32 32 3];
imgStat   = [0.485, 0.456, 0.406; % mean
             0.229, 0.224, 0.225];% std
net.seed  = seed;         
%% Neural Network properties
net.task           = 'classification';
net.modelName      = modelName;
net.dataName       = dataName;
net.cd             = cd;
% Save net after N epoch
net.maxEpoch       = 10; 
net.savedEpoch     = round(net.maxEpoch/2);
% GPU 1: yes; 0: no
net.gpu            = true;
net.cuda           = true;% run cuda compiler version?
net.numDevices     = 1;
% Data type object half or double precision
net.dtype          = 'single';
% Number of input covariates
net.imgSize        = imgSize;
net.resizeMode     = false;
net.resizedSize    = nan;
net.nx             = prod(net.imgSize); 
net.imgStat        = imgStat;
% Number of output responses
ny                 = trainImdb.numOutputs; 
net.ny             = ny;   
net.nl             = ny;
net.nv2            = ny; 
net.nye            = trainImdb.numEnOutputs; % Num. of hidden states
% Number of classes
net.numClasses     = trainImdb.numClasses;
% Batch size 
net.batchSize      = 10; 
net.repBatchSize   = 1;
% Number of nodes for each layer| 1: FC; 2:conv; 3: pooling
net.imgSize        = imgSize;
net                = indices.layerEncoder(net);
[net.layer, net.filter, net.kernelSize, net.padding, net.paddingType, net.stride, net.nodes, net.actFunIdx, net.xsc] = resnet.v1(imgSize, net.ny, 3);
net.actBound       = ones(size(net.layer));
% Observations standard deviation
net.sv             = 4*ones(1,1);      
net.svDecayFactor  = 0.975;
net.svmin          = 0.3;
net.lastLayerUpdate = true; 

% Misc
net.isUdidx        = true;
net.errorRateEval  = true;

% Factor for initializing weights & bias | {Xavier, He}
net.initParamType  = 'He';
% Data augmentation
% net                = indices.daTypeEncoder(net);
% net.da.enable      = false;
% net.da.p           = [0.5, 1];
% net.da.types       = [net.da.horizontalFlip, net.da.randomCrop];  
% net.da.randomCropPad = 4;
net.obsShow        = 500;

%% Pretrained model directory
trainedModelDir = []; 
initModelDir    = [];
initEpoch = 0;

%% Run
task.runClassification(net, trainImdb , testImdb , trainedModelDir, initModelDir, initEpoch);

