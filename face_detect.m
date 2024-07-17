function [Face, output] = face_detect(Img)

%face landmark detection using MTCNN
%https://github.com/matlab-deep-learning/mtcnn-face-detection

detector = mtcnn.Detector();

[bboxes, scores, landmarks] = detector.detect(Img);

if isempty(bboxes)
    Face=0;
    return;
else
    idx = 1;
    if size(bboxes,1)>1
        %bboxes=bboxes(size(bboxes,1),:);
        [~,idx]=max(bboxes(:,3).*bboxes(:,4));
        bboxes = bboxes(idx,:);
    end
    
    output{1} = bboxes;
    % landmark in (idx,5,2) format for idx be number of faces, 5 for num of landmarks, 2 for x and y
    Face=imcrop(Img,bboxes);
    left_eye = squeeze(landmarks(idx, 1, :));
    right_eye = squeeze(landmarks(idx, 2, :));

    aspect_ratio = 96 / 64;
    eye_distance = norm(left_eye - right_eye);
    eye_bbox_width = eye_distance * 1;  % Adjust as necessary
    eye_bbox_height = eye_bbox_width / aspect_ratio;

    left_eye_bbox_x = left_eye(1) - eye_bbox_width / 2;
    left_eye_bbox_y = left_eye(2) - eye_bbox_height / 2;
    
    right_eye_bbox_x = right_eye(1) - eye_bbox_width / 2;
    right_eye_bbox_y = right_eye(2) - eye_bbox_height / 2;

    left_eye_bbox = [left_eye_bbox_x, left_eye_bbox_y, eye_bbox_width, eye_bbox_height];
    right_eye_bbox = [right_eye_bbox_x, right_eye_bbox_y, eye_bbox_width, eye_bbox_height];

    output{2} = left_eye_bbox;
    output{3} = right_eye_bbox;
    
end
end