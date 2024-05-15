clc 
clear

% Load the BostonHousing dataset
data = readtable('BostonHousing.csv');

% Split the data into predictors (X) and response variable (y)
X = table2array(data(:, 1:13));
y = table2array(data(:, 14));

% Split the data into training and testing sets
rng(1); % For reproducibility
cv = cvpartition(size(X,1),'HoldOut',0.3);
idxTrain = training(cv);
XTrain = X(idxTrain,:);
yTrain = y(idxTrain,:);
XTest  = X(~idxTrain,:);
yTest  = y(~idxTrain,:);

% Train a linear regression model
mdl = fitlm(XTrain, yTrain);

% Predict on the test set
yPred = predict(mdl, XTest);

% Evaluate the model
mse = mean((yTest - yPred).^2);
rmse = sqrt(mse);
fprintf('Root Mean Squared Error: %.2f\n', rmse);

% Plot actual vs. predicted values
figure;
scatter(yTest, yPred);
hold on;
plot([min(yTest), max(yTest)], [min(yTest), max(yTest)], 'r--');
xlabel('Actual Values');
ylabel('Predicted Values');
title('Actual vs. Predicted Values');

% Plot histogram of residuals
residuals = yTest - yPred;
figure;
histogram(residuals, 'Normalization', 'probability');
xlabel('Residuals');
ylabel('Probability');
title('Histogram of Residuals');
