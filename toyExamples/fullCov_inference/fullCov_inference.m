%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:         fullCov_inference
% Description:  test observation error full covariance inference
% Authors:      Luong-Ha Nguyen, James-A. Goulet & Ali Fakhri
% Created:      June 28, 2022
% Updated:      July 7, 2022
% Contact:      luongha.nguyen@gmail.com, james.goulet@polymtl.ca & afakhri@pm.me
% Copyright (c) 2022 Luong-Ha Nguyen, James-A. Goulet & Ali Fakhri. All rights reserved. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ========================================================================
% Generate data
seed = 420;
a = -0.75; b = 0.75; % sampling interval
n_samples = 2000;
n_out = 3;
[xtrain, ytrain, ~] = generate_data(a, b, n_samples, n_out, seed, false);
[xtest, ytest, error_covs] = generate_data(a, b, n_samples, n_out, seed, true);

% ========================================================================
% Set NN attributes
net.task           = 'regression';
net.saveModel      = 0;
net.maxEpoch       = 30;
% GPU
net.gpu            = false;
net.cuda           = false;
net.numDevices     = 1;
% Data type object half or double precision
net.dtype          = 'single';
% Number of input covariates
net.nx             = size(xtrain, 2); 
% Number of output responses
net.nl             = size(ytrain, 2);
net.nv2            = size(ytrain, 2);
net.nLchol         = (net.nv2^2 + net.nv2)/2;
net.ny             = net.nl + net.nLchol; 
% Batch size 
net.batchSize      = 1; 
net.repBatchSize   = 1;

% Noise parameter
net.sv = 0;
net.learnSv = 1;
net.noiseType = 'full';

% Build the network architecture
% Layer| 1: FC; 2:conv; 3: max pooling; 4: avg pooling; 5: layerNorm; 6: batchNorm 
% Activation: 1:tanh; 2:sigm; 3:cdf; 4:relu; 5:softplus
net.layer          = [1         1       1     ];
net.actFunIdx      = [0         4       0     ];

% Parameter initialization
net.initParamType  = 'He';
numLayers = length(net.layer);

% grid search values
gain_values = [0.01, 0.1, 0.5, 0.75, 1];
hidden_nodes = [50, 100, 200, 400];
gain_max = 1;
gain_min = 1e-2;
h_nodes_max = 500;
h_nodes_min = 50;

config = {};
grid_search = false;
randomized_search = false;
total_attempts = 1; % change to value > 1 if doing gridsearch or randomized search
attempts = 0;
best_LL = -inf;

while attempts < total_attempts
    %     seed = randi(1e8);
    if ~isempty(seed)
        rng(seed);
        net.seed = seed;
        fprintf('---------------------------------\n')
        fprintf('Seed: %d\n', seed);
        config.seed = seed;
    else
        rng('shuffle');
        rng_ = rng;
        fprintf('---------------------------------\n')
        fprintf('Rng: %d\n', rng_.Seed);
        config.rng = rng_;
    end

    if grid_search
        mw_gain = randsample(gain_values,1);
        mb_gain = randsample(gain_values,1);
        Sw_gain = randsample(gain_values,1);
        Sb_gain = randsample(gain_values,1);
        h_nodes = randsample(hidden_nodes,1);
    elseif randomized_search
        mw_gain = (gain_max-gain_min).*rand + gain_min;
        mb_gain = (gain_max-gain_min).*rand + gain_min;
        Sw_gain = (gain_max-gain_min).*rand + gain_min;
        Sb_gain = (gain_max-gain_min).*rand + gain_min;
        h_nodes = randi([h_nodes_min, h_nodes_max]);
    else % use the default values
        mw_gain = 1;
        mb_gain = 1;
        Sw_gain = 0.75;
        Sb_gain = 0.75;
        h_nodes = 400;
    end

    net.gainMw = mw_gain.*ones(1, numLayers-1);
    net.gainSw = Sw_gain.*ones(1, numLayers-1);
    net.gainMb = mb_gain*ones(1, numLayers-1);
    net.gainSb = Sb_gain*ones(1, numLayers-1);
    net.nodes = [net.nx    h_nodes     net.ny]; 
    
    % save the config
    config.mw_gain = mw_gain;
    config.mb_gain = mb_gain;
    config.Sb_gain = Sb_gain;
    config.Sw_gain = Sw_gain;
    config.layers = net.nodes;

    % ========================================================================
    % Train the  model
    patience = 0;
    patience_threshold = 5;
    
    % Train net
    net.trainMode = 1;
    [net, states, maxIdx, netInfo] = network.initialization(net);
    normStat = tagi.createInitNormStat(net);
    
    % Validation net
    netT              = net;
    netT.trainMode    = 0;
    [netT, statesT, maxIdxT] = network.initialization(netT); 
    normStatT = tagi.createInitNormStat(netT);
    
    theta = tagi.initializeWeightBias(net);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Training
    stop  = 0;
    epoch = 0;
    best_epoch = 0;
%     tic
    while ~stop
        fprintf('\nEpoch: %d \n==============\n', epoch)
        
        if epoch >1
            idxtrain = randperm(size(ytrain, 1));
            ytrain   = ytrain(idxtrain, :);
            xtrain   = xtrain(idxtrain, :);
        end
    
        [theta, normStat, mytrain, Sytrain,...
            ~] = network.regression(net, theta, normStat, states,...
            maxIdx, xtrain, ytrain);
        
        if any(isnan(mytrain))
            fprintf('\nWarn! NaN values in predicted mean\n');
        end
        if any(isnan(Sytrain))
            fprintf('Warn! NaN values in oredicted variance\n');
        end
    
        % Validating
        [~, ~, myvalid, Syvalid] = network.regression(netT, theta, ...
        normStatT, statesT, maxIdxT, xvalid);
    
        mv2_test = zeros(size(myvalid(:,net.nl+1:end)));
        Sv2_test = zeros(size(Syvalid(:,net.nl+1:end)));
        
        for i=1:size(myvalid,1)
            ma = myvalid(i,:);
            Sa = Syvalid(i,:);
            [~, mLa] = tagi.detachMeanVar(ma, net.nl,...
                net.nLchol, net.batchSize, net.repBatchSize);
            [~, SLa] = tagi.detachMeanVar(Sa, net.nl,...
                net.nLchol, net.batchSize, net.repBatchSize);
            % transform the diagonal elements into positive domain
            [mLa_, SLa_, ~] = agvi.transform_chol_vec(mLa, SLa, net.gpu);
            % retrieve the variance parameters from the cholesky vectors
            [mv2, Sv2] = agvi.chol_to_v2(mLa_, SLa_);
            mv2_test(i,:) = mv2;
            Sv2_test(i,:) = diag(Sv2);
        end
        
        LL_val = mt.loglikfull(yvalid, myvalid(:,1:net.nl), Syvalid(:,1:net.nl), mv2_test);

        if isnan(LL_val)
            LL_val = -inf;
        end
        fprintf('Valid L.L. : %d \n',LL_val)
    
        if LL_val > best_LL
            best_LL = LL_val;
            best_theta = theta;
            best_epoch = epoch;
            patience = 0;
            best_config = config;
            best_config.LL = best_LL;
        else
            patience = patience + 1;
        end
        
        epoch = epoch + 1;
        
        if epoch > net.maxEpoch
            fprintf('Epoch > %d, max epoch exceeded. Training terminated... best epoch: %d \n', net.maxEpoch, best_epoch)
            stop = true;
        elseif patience == patience_threshold
            fprintf('Early stopping... Training terminated at epoch %d, best epoch: %d \n', epoch-1, best_epoch)
            stop = true;
        end
    end
    attempts = attempts + 1;
end
% toc
fprintf('************************************\n')
% fprintf('\nBest seed: %d', best_seed)
fprintf('\nBest LL_val: %d', best_LL)

% ========================================================================
% Predict
[~, ~, pred_mean, pred_var] = network.regression(netT, best_theta, ...
    normStatT, statesT, maxIdxT, xtest, []);

% Transform from the 'Cholesky space' to the 'variance space'
mv2_test = zeros(size(pred_mean(:,net.nl+1:end)));
Sv2_test = zeros(size(pred_mean(:,net.nl+1:end)));

for i=1:size(pred_mean,1)
    ma = pred_mean(i,:);
    Sa = pred_var(i,:);
    [~, mLa] = tagi.detachMeanVar(ma, net.nl,...
        net.nLchol, net.batchSize, net.repBatchSize);
    [~, SLa] = tagi.detachMeanVar(Sa, net.nl,...
        net.nLchol, net.batchSize, net.repBatchSize);
    % transform the diagonal elements into positive domain
    [mLa_, SLa_, ~] = agvi.transform_chol_vec(mLa, SLa, net.gpu);
    % retrieve the variance parameters from the cholesky vectors
    [mv2, Sv2] = agvi.chol_to_v2(mLa_, SLa_);
    mv2_test(i,:) = mv2;
    Sv2_test(i,:) = diag(Sv2);
end

% ========================================================================
% Plot the predicted target variables
var_indices = [1, 3, 6];
for i=1:net.nl
    figure;
    pl.regression(xtest, pred_mean(:,i), pred_var(:,i) + mv2_test(:,var_indices(i)), 'black', 'green', 1)
    hold on
    scatter([xtrain;xvalid], [ytrain(:,i);yvalid(:,i)], 10, 'magenta', 'o', 'DisplayName','training data')
    hold on 
    plot(xtest, ytest, 'red', 'DisplayName','true')
end

% ========================================================================
% Plot the predicted variances and covariances

% Obtain indices for an upper triangular matrix of size (m,m)
m_dim = floor(sqrt(2*net.nLchol));
triu_mat = triu(ones(m_dim));
[triu_ind_i, triu_ind_j] = ind2sub(size(triu_mat), find(logical(triu_mat)));

for i=1:net.nLchol
    row_ind = triu_ind_i(i);
    col_ind = triu_ind_j(i);
    if row_ind == col_ind
        label = strcat('$\sigma_{',num2str(row_ind),'}^2$ true');
    else
        label = strcat('$\sigma_{',strcat(num2str(row_ind),num2str(col_ind)),'}$ true');
    end
    figure;
    pl.regression(xtest, mv2_test(:,i), Sv2_test(:,i), 'black', 'green', 1)
    hold on
    plot(xtest, error_covs(:,row_ind,col_ind),'DisplayName',label)
%     xlim([-0.75 0.75]) % limit the plot to the to the range of the training data
end

% ========================================================================
% Function for generating data
function [x_all, y_all, error_covs] = generate_data(a, b, n_samples, n_out, seed, test)
    if ~isempty(seed)
        rng(seed);
    end

    if test
        x_all = linspace(a,b)';
    else
        x_all = (b - a) * rand(n_samples, 1) + a;
    end

    errors = zeros(size(x_all,1), n_out);
    error_covs = zeros(size(x_all,1), n_out, n_out);

    for i=1:size(x_all,1)
        x = x_all(i);
    
        sigma1 = sqrt(0.1 * x + 0.25);
        sigma2 = sqrt(0.5 * x^2 + 0.1);
        sigma3 = sqrt(0.5 * (x + 1.5)^2);
    
        diag_std = diag([sigma1, sigma2, sigma3]);
        rhos = [0.1, -0.3, -0.25];

        corr_matrix = ones(n_out);
        corr_matrix(logical(triu(corr_matrix,1))) = rhos;
        corr_matrix(logical(triu(corr_matrix,1)')) = rhos;
    
        error_cov_matrix = diag_std * corr_matrix * diag_std;

        error_covs(i,:,:) = error_cov_matrix;
        errors(i,:) = mvnrnd(zeros(n_out,1), error_cov_matrix);
    end

    if test
        y_all = (1 + x_all) .* sin(x_all * pi) .^ 3;
    else
        y_all = (1 + x_all) .* sin(x_all * pi) .^ 3 + errors;
    end
end