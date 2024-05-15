clc 
clear

% Step 1: Load and Preprocess Data
data = readtable('iris.csv'); % Load Iris dataset from CSV file
X = table2array(data(:,1:4)); % Features
Y = data.variety; % Extract labels from table variable
Y = categorical(Y); % Convert labels to categorical array

% Convert categorical labels to numeric matrix (one-hot encoding)
YNumeric = dummyvar(double(Y));

% Step 2: Split Data into Training and Test Sets
cv = cvpartition(height(data), 'HoldOut', 0.3);
XTrain = X(training(cv), :);
YTrainNumeric = YNumeric(training(cv), :);
XTest = X(test(cv), :);
YTest = Y(test(cv));

% Step 3: Create and Train Neural Network Model
net = patternnet(10); % Create a neural network with 10 hidden units
net = train(net, XTrain', YTrainNumeric'); % Train the model with numeric targets

% Step 4: Evaluate Model Performance
YTestPred = net(XTest');
[~, YTestPredIdx] = max(YTestPred);
[~, YTestIdx] = max(YTest, [], 2);
accuracy = sum(YTestPredIdx == YTestIdx) / numel(YTestIdx);
confMat = confusionmat(YTestIdx, YTestPredIdx);
disp(['Accuracy: ' num2str(accuracy)]);
disp('Confusion Matrix:');
disp(confMat);

% Step 5: Visualization (Optional)
plotconfusion(YTestIdx, YTestPredIdx); % Plot confusion matrix

% Step 6: Save Trained Model (Optional)
save('iris_neural_network_model.mat', 'net'); % Save trained model