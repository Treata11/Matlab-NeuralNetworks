clear 
clc

% 1. We load the weather data from the CSV file into a table.
% 2. We select relevant features (Data.Precipitation, Data.Wind.Direction, Data.Wind.Speed) and the target variable (Data.Temperature.Avg Temp).
% 3. We normalize the input features.
% 4. We create and train an RBF network with 10 hidden neurons using the newrb function.
% 5. We make a prediction for a new data point [0.1, 45, 5] by normalizing it and using the trained network.

%% 

% Load the weather data from the CSV file
data = readtable('weather.csv');

% Select relevant features for training the RBF network
X = data{:, {'Data_Precipitation', 'Data_Wind_Direction', 'Data_Wind_Speed'}};
y = data.Data_Temperature_AvgTemp;

% Normalize the input features
X = normalize(X);

% Create and train the RBF network
hiddenSize = 10; % Number of hidden neurons
net = newrb(X', y', 0, 1, hiddenSize);

% Predict the average temperature for a new data point
newDataPoint = [0.1, 45, 5]; % Example new data point
normalizedDataPoint = (newDataPoint - mean(X))./std(X); % Normalize the new data point
predictedTemp = sim(net, normalizedDataPoint'); % Predict the average temperature

disp(['Predicted Average Temperature: ', num2str(predictedTemp)]);
