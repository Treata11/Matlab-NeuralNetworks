clear
clc 

% Load the weather dataset
data = readtable('weather.csv');

% Preprocess the data
X = data{:, 10:end}; % Selecting temperature and wind data as features
y = data{:, 1}; % Selecting precipitation as the target variable

% Normalize the data
X = normalize(X);
y = normalize(y);

% Split the data into training and testing sets
X_train = X(1:500, :);
y_train = y(1:500);
X_test = X(501:end, :);
y_test = y(501:end);

% Define the RNN model
layers = [ ...
    sequenceInputLayer(size(X_train, 2))
    lstmLayer(100)
    fullyConnectedLayer(1)
    regressionLayer];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.01, ...
    'Plots', 'training-progress');

% Train the RNN model
net = trainNetwork(X_train', y_train', layers, options);

% Make predictions on test data
predictions = predict(net, X_test');

% Plot the actual vs predicted values
figure;
plot(y_test);
hold on;
plot(predictions);
legend('Actual', 'Predicted');
title('Actual vs Predicted Precipitation');

