clc 
clear

% Load the Iris dataset
irisData = readtable('Iris.csv');

% Extract features and labels
X = table2array(irisData(:,1:4)); % Features
Y = grp2idx(irisData{:,5}); % Labels

% Split the dataset into training and testing sets
cv = cvpartition(size(X,1),'HoldOut',0.3);
idxTrain = training(cv); % Training indices
XTrain = X(idxTrain,:);
YTrain = Y(idxTrain,:);
XTest = X(~idxTrain,:);
YTest = Y(~idxTrain,:);

% Train the ECOC classifier
ECOCModel = fitcecoc(XTrain,YTrain);

% Predict on the test set
YTestPredicted = predict(ECOCModel,XTest);

% Evaluate the performance
accuracy = sum(YTestPredicted == YTest) / numel(YTest);
disp(['Accuracy: ', num2str(accuracy)]);


% FIXME: See what's wrong with the plot
%% Create a confusion matrix
C = confusionmat(YTest, YTestPredicted);

% Display the confusion matrix
figure;
confusionchart(C, {'Setosa', 'Versicolor', 'Virginica'});
title('Confusion Matrix');

% Create a classification report
classLabels = unique(Y);
classNames = {'Setosa', 'Versicolor', 'Virginica'};
classLabelsStr = arrayfun(@num2str, classLabels, 'UniformOutput', false);
classificationReport = table(classNames', C, 'VariableNames', {'Class', 'Accuracy', 'Precision', 'Recall'});

disp('Classification Report:');
disp(classificationReport);