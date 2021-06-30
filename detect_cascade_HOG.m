function [chosen_idx, score] = detect_cascade_HOG(cascade_svm, cascade_th, total_wins)
  n_level = length(cascade_svm);

  n_wins = length(total_wins);
  chosen_idx = 1:n_wins;
  chosen_wins = total_wins;

  for i=1:n_level
    block_pos = cascade_svm{i}(:, 1:4); % [x, y, w, h]
    betas = cascade_svm{i}(:,5:end-3);
    biases = cascade_svm{i}(:,end-2);
    alphas = cascade_svm{i}(:,end-1);
    round_th = cascade_svm{i}(:,end);
    level_th = cascade_th{i};
    
    n_filtered_wins = length(chosen_wins);
    n_round = size(alphas,1);
    score = zeros(n_filtered_wins,1);
    
    for j=1:n_round
      position = block_pos(j,:);
      total_blocks = [];
      for k=1:n_filtered_wins
        rc = xy_to_rc(position);
        block = chosen_wins{k}(rc(1):rc(3),rc(2):rc(4));
        total_blocks = [total_blocks block];
      end
      X_test = extractHOGFeatures(total_blocks,'CellSize', [position(4)/2 position(3)/2], 'BlockOverlap', [0 0]);
      X_test = reshape(X_test, 36, n_filtered_wins);
      X_test = X_test';
      round_label = (X_test * (betas(j,:))' + biases(j)) >= round_th(j);
      score = score + round_label.*alphas(j);
    end
    
    idx = find(score >= level_th);
    if size(idx,1) == 0
      break;
    end
    
    score = score(idx);
    chosen_wins = chosen_wins(idx);
    chosen_idx = chosen_idx(idx);
  end

end
