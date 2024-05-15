clear 
clc

% FIXME: Does not compile
%% Convolutional Neural Network

% Load the weather dataset
data = readtable('weather.csv');

% Extract relevant features and labels
X = [data.Data_Precipitation, data.Data_Temperature_AvgTemp, data.Data_Temperature_MaxTemp, data.Data_Wind_Speed];
Y = categorical(data.Station_State); % Assuming 'Station.State' is the target variable

% Convert categorical labels to numeric indices
Y_indices = grp2idx(Y);

% Convert numeric indices to categorical format
Y_categorical = categorical(Y_indices);

% Reshape input data for CNN
X = permute(X, [3, 1, 2]); % Reshape to [1, 4, N] where N is the number of samples

% Create the CNN model (same as before)
layers = [
    imageInputLayer([1 4 1])
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([1,2],'Stride',2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',10, ...
    'MiniBatchSize',32, ...
    'InitialLearnRate',0.001);

% Train the CNN model with categorical labels
net = trainNetwork(X, Y_categorical, layers, options); % Use Y_categorical instead of Y_indices

% Make predictions using the trained model (you can split the data into training and testing sets for better evaluation)
predictions = classify(net, X);

% Display confusion matrix
figure;
plotconfusion(Y_categorical, predictions);
