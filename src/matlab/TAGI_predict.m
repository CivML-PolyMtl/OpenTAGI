function [mytest, Sytest] = TAGI_predict(net, theta, x_test)
    % Test net
    netT              = net;
    netT.trainMode    = 0;
    netT.batchSize    = 1;
    netT.repBatchSize = 1;
    [netT, statesT, maxIdxT] = network.initialization(netT); 
    normStatT = tagi.createInitNormStat(netT);

    [~, ~, mytest, Sytest] = network.regression(netT, theta, ...
    normStatT, statesT, maxIdxT, x_test, []);
end