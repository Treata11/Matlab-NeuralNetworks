clc 
clear

% Step 1: Data Preparation
[XTrain, YTrain, XTest, YTest] = loadMNISTData(); % Custom function to load MNIST dataset
XTrain = normalizeData(XTrain);
XTest = normalizeData(XTest);

% Step 2: Network Creation
hiddenLayerSize = 100;
net = feedforwardnet(hiddenLayerSize);
net.layers{1}.transferFcn = 'logsig'; % Sigmoid activation function for hidden layer
net.layers{2}.transferFcn = 'softmax'; % Softmax activation function for output layer

% Step 3: Training
net.trainParam.epochs = 20;
net.trainParam.lr = 0.01;
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:50000;
net.divideParam.valInd = 50001:60000;
net.divideParam.testInd = [];
net = train(net, XTrain', YTrain');

% Step 4: Performance Evaluation
YTestPred = net(XTest');
accuracy = sum(YTestPred == YTest) / numel(YTest);
confMat = confusionmat(YTest, YTestPred);
plotconfusion(YTest, YTestPred); % Plot confusion matrix

% Step 5: Hyperparameter Tuning (Optional)
% Perform hyperparameter tuning using grid search or Bayesian optimization

% Step 6: Transfer Learning (Optional)
% Fine-tune a pre-trained neural network model for digit recognition

% Step 7: Deployment
save('handwritten_digit_recognition_model.mat', 'net'); % Save trained model

% Step 8: Visualization
plotTrainingResults(net); % Custom function to plot training progress and results
