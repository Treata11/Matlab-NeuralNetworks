clear 
clc

%% Requires Vision toolbox

% Read the image file
img = imread('Face.png');

% Use the Viola-Jones face detection algorithm to detect the face
faceDetector = vision.CascadeObjectDetector;
bbox = step(faceDetector, img);

% Draw bounding box around the detected face
imgWithFace = insertObjectAnnotation(img, 'rectangle', bbox, 'Face');
figure;
imshow(imgWithFace);
title('Detected Face');

% Use a pre-trained deep learning model for facial feature recognition
faceImage = imcrop(img, bbox(1,:)); % Crop the detected face region
net = alexnet; % Use a pre-trained CNN model for feature recognition
predictions = classify(net, faceImage);

% Display the original image with bounding boxes around the detected facial features
figure;
imshow(img);
% hold on;

% Draw bounding boxes around the detected facial features
% for i = 1:numel(predictions)
%     label = predictions(i);
%     % You can use a more accurate method to detect specific facial features here
%     % For example, you can use additional pre-trained models or custom algorithms
% 
%     % Example: Draw a rectangle around the detected feature
%     bboxFeature = [50, 50, 100, 100]; % Example bounding box coordinates
%     rectangle('Position', bboxFeature, 'EdgeColor', 'r', 'LineWidth', 2);
%     text(bboxFeature(1), bboxFeature(2) - 10, label, 'Color', 'r', 'FontSize', 12);
% end

% title('Facial Feature Detection');
