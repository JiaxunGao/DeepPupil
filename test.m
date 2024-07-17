clear;
clc;
detector = mtcnn.Detector();

% Load the trained DeepPupilNet eye localization network
load('DeepPupilNet.mat');

% Find sample images
cd Images
d = dir;
d = d(3:end);
cd ..

for i=1:length(d)
    
    % Read every sample image
    cd Images
    Im = imread(d(i).name);
    cd ..
    
    % Estimate the eye center coordinates
    [bboxes, scores, landmarks] = detector.detect(Im);

    %print number of faces detected
    fprintf('Detected %d faces\n', numel(scores));
    disp(scores)
    disp(landmarks);
    
    % plot bonding box
    figure, imshow(Im), hold on;
    
    for j = 1:size(bboxes, 1)
        rectangle('Position', bboxes(j, :), 'EdgeColor', 'r', 'LineWidth', 2);
    end
    for iFace = 1:numel(scores)
        scatter(landmarks(iFace, 1, 1), landmarks(iFace, 1, 2), 'filled');
    end
    hold off;
    
end
