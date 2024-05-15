clc 
clear

% Read the Iris dataset from a CSV file into a table
irisTable = readtable('Iris.csv');

% Extract input features and target labels
X = irisTable{:, 1:4}; % Input features
Y = irisTable.variety; % Target labels

% Convert target labels to numeric indices
[~, Y_idx] = ismember(Y, unique(Y));

% Convert target labels to categorical array
Y_categorical = categorical(Y_idx);

% Create a feedforward neural network model
hiddenLayerSize = 10;
net = feedforwardnet(hiddenLayerSize);

% Train the neural network
net = train(net, X', ind2vec(Y_idx'));

% Make predictions using the trained network
predictions = net(X');

% Convert predicted indices back to categorical labels
predicted_labels = categories(Y);
predicted_labels = predicted_labels(vec2ind(predictions));

% Display the confusion matrix
plotconfusion(Y, predicted_labels);

% Plot the actual target values and predicted values
subplot(2,1,1);
plot(Y_categorical);
title('Actual Target Values');
xlabel('Sample Index');
ylabel('Class');

subplot(2,1,2);
plot(predictions);
title('Predicted Values');
xlabel('Sample Index');
ylabel('Class');