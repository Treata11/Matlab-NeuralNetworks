clear
clc

%% Recurrent Neural Networks (RNN)

% Load the weather dataset
data = readtable('weather.csv');

% Extract relevant features and labels
X = [data.Data_Precipitation, data.Date_Month, data.Date_WeekOf, data.Date_Year, data.Data_Wind_Direction, data.Data_Wind_Speed];
Y = data.Data_Temperature_AvgTemp;

% Preprocess the data
mu = mean(X);
sigma = std(X);
X_normalized = (X - mu) ./ sigma;

% Split the data into training and testing sets
split_idx = floor(0.8 * size(X, 1));
X_train = X_normalized(1:split_idx, :);
Y_train = Y(1:split_idx);
X_test = X_normalized(split_idx+1:end, :);
Y_test = Y(split_idx+1:end);

% Create the RNN model
inputSize = size(X, 2);
numHiddenUnits = 100;
numResponses = 1; % Since we are predicting a single value (average temperature)
layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numResponses)
    regressionLayer];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'MiniBatchSize',32, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','every-epoch', ...
    'Verbose',0);

% Train the RNN model
net = trainNetwork(X_train', Y_train', layers, options);

% Make predictions using the trained model
YPred = predict(net, X_test');

% Evaluate the performance of the model
rmse = sqrt(mean((YPred - Y_test').^2));
fprintf('Root Mean Squared Error (RMSE): %.2f\n', rmse);
