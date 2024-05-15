clc 
clear

%% Unsupervised Learning

% Load the Iris dataset
iris = readtable('iris.csv');
X = table2array(iris(:, 1:4));

% Perform k-means clustering with k=3 
% Play around with k 
rng(1); % For reproducibility
[idx, C] = kmeans(X, 3);

% Plot the clustered data
figure;
gscatter(X(:,1), X(:,2), idx, 'rgb', 'o', 8);
hold on;
plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
xlabel('Sepal Length');
ylabel('Sepal Width');
title('K-means Clustering of Iris Dataset');

% Add legend
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids');

% Check the quality of clustering using silhouette plot
silhouette(X, idx);
