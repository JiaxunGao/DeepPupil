clear;
clc;

% Load the trained DeepPupilNet eye localization network
load('DeepPupilNet.mat');

% Video file path
videoFile = 'video.mp4'; % Specify the path to your video file

% Create a VideoReader object to read the video
video = VideoReader(videoFile);

% Initialize an array to store pupil coordinates
numFrames = floor(video.Duration * video.FrameRate);
pupil_coords = zeros(numFrames, 4);

frameIndex = 1;

while hasFrame(video)

    fprintf('Processing frame %d\n', frameIndex);
    Im = readFrame(video);
    
    [Coords, bboxes] = eye_localization(Im, net);

    %check whether the coords contains 0 at any index
    if any(Coords == 0)
        fprintf('Face not found\n');
        %repeat the coords to be the same as the previous frame
        Coords = pupil_coords(frameIndex - 1, :);
    end

    pupil_coords(frameIndex, :) = Coords;
    
    % Display the results
    %figure, imshow(Im), hold on;
    %plot(Coords(1), Coords(2), 'Marker', '+', 'LineWidth', 1.5);
    %plot(Coords(3), Coords(4), 'Marker', '+', 'LineWidth', 1.5);
    %hold off;
    
    frameIndex = frameIndex + 1;
end

%remove (0,0,0,0) rows from pupil_coords
pupil_coords = pupil_coords(any(pupil_coords, 2), :);

pupil_coords_table = array2table(pupil_coords, ...
    'VariableNames', {'Eye1_X', 'Eye1_Y', 'Eye2_X', 'Eye2_Y'});

%plot x coordinate of pupil
%adjust the size of the figure
figure('Position', [100, 100, 800, 400]);
plot(pupil_coords(:,1), 'r', 'LineWidth', 1.5);
hold on;
plot(pupil_coords(:,3), 'b', 'LineWidth', 1.5);
xlabel('Frame Index');
ylabel('X Coordinate');
legend('Eye 1', 'Eye 2');
title('Pupil X Coordinates Over Time');
hold off;
% Save the table as a CSV file
writetable(pupil_coords_table, 'pupil_coordinates.csv');
