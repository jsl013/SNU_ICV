function test_cascade_HOG(cascade_svm, cascade_th, test_pos_path, scale, win_size, step_size, label_path)
  pos = 1;
  padding = [0 0];
  overlap_th = 0.3;
  
  [pos_wins, pos_position, ~] = extract_multiscale_windows(test_pos_path, pos, scale, padding, win_size, step_size);

  [chosen_idx, score] = detect_cascade_HOG(cascade_svm, cascade_th, pos_wins);
  chosen_pos = pos_position(chosen_idx,:);
  boxes = [chosen_pos score];
  n_img = size(unique(boxes(:,2)),1);
  pick = [];
  for i=1:n_img
    img_boxes = boxes(boxes(:,2)==i,:);
    pick = [pick ; nms(img_boxes, overlap_th)];
  end
  save(sprintf('cascade_hog_pick_%d', n_last_img), 'pick');
  pos_lists = dir(test_pos_path);
  pos_lists = pos_lists(3:end);
  n_pick = size(pick,1);
  image_ids = cell(n_pick,1);
  for i=1:n_pick
    image_ids{i} = pos_lists(pick(i,2)).name; % img name
  end
  refined_position = pick(:,3:end-1); % [x, y, w, h]
  refined_position(:,3) = pick(:,3) + pick(:,5) -1;
  refined_position(:,4) = pick(:,4) + pick(:,6) -1;
  confidences = pick(:,end);
  
  [~, ~, ~, tp, fp, ~] = evaluate_detections(refined_position, confidences, image_ids, label_path, 1);
  visualize_detections_by_image(refined_position, confidences, image_ids, tp, fp, test_pos_path, label_path);
end
