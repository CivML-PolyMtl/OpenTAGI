%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         visualField_lv_V1
% Description:  test visual field on mnist
% Authors:      Luong-Ha Nguyen & James-A. Goulet
% Created:      March 16, 2021
% Updated:      March 16, 2021
% Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
% Copyright (c) 2021 Luong-Ha nguyen & James-A. Goulet 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
% clc
% close all
set(0,'DefaultAxesFontName','Helvetica')
set(0,'defaultLineLineWidth',2)
set(0,'DefaultAxesFontSize',14)
set(0,'defaulttextfontsize',14)
format shortE
rng(0)

%% Data
modelName    = 'visualField_lv_N5_V1';
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
trainIdx4val = 1:50000;
valIdx4val   = (length(trainIdx4val)+1):length(trainIdx);
x            = single(x);
x            = (x-mean(reshape(x, [numel(x), 1])))/std(reshape(x, [numel(x), 1]));
imgSize      = [28 28 1];
% x            = dp.cvr2img(x, imgSize);
y            = single(y(idxN));
yref         = y;
nx           = prod(imgSize); 
% x            = x(:,:,:, idxN);
numberRef    = 3;
idx          = y==numberRef;
x            = x(idx, :);
x            = x(1:10, :);

%% Coordinates
z            = linspace(-1, 1, imgSize(1));
[zx, zy]     = ndgrid(z, z);
zx           = reshape(zx, [numel(zx), 1]);
zy           = reshape(zy, [numel(zy), 1]);
z            = [zx, zy];

%% Net
net.task           = 'classification';
net.modelName      = modelName;
net.dataName       = dataName;
net.cd             = cd;
% Save net after N epoch
net.maxEpoch       = 20; 
net.savedEpoch     = round(net.maxEpoch/2);
% GPU 1: yes; 0: no
net.gpu            = 0;
net.numDevices     = 1;
% Data type object half or double precision
net.dtype          = 'single';
% Encoding
[y,net.encoderIdx] = dp.encoder(y, Nclass, net.dtype);      
ny                 = 1; 
net.labels         = yref;
% Number of input covariates
net.numCoord       = 2;
net.numlv          = 2;
net.nx             = net.numCoord  + net.numlv; 
% Number of output responses   
net.nl             = ny;
net.nv2            = ny; 
net.ny             = ny;
% Number of classes
net.numClasses     = Nclass;
% Batch size 
net.batchSize      = 16; 
net.repBatchSize   = 1;

% Layer| 1: FC; 2:conv; 3: max pooling; 
% 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.imgSize        = imgSize;
net.layer          = [1         1      1   1];
net.nodes          = [net.nx    50     50 net.ny]; 
net.actFunIdx      = [0         4      4   0];
net.actBound       = [1         1      1   1];
% Observations standard deviation
net.learnSv        = false; % Online noise learning
net.sv             = 0.5*ones(1,1, net.dtype);      
% Parameter initialization
net.initParamType  = 'Xavier';
% Update type
net.lastLayerUpdate = true; 
net.collectDev      = false;
net.convariateEstm  = true;
% Transfer to GPU
if net.gpu == 1
    x = gpuArray(x);
    net.encoderIdx = gpuArray(net.encoderIdx);
    y = gpuArray(y);
end 

%% Pretrained model directory
trainedModelDir = []; 
initModelDir = [];
initEpoch = 0;
%% Run
task.runImageGenerationVFtrain_V2(net, z, x, trainedModelDir, initModelDir, initEpoch)


