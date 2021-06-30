function [cascade_svm, cascade_th] = train_cascade_HOG(pos_windows, neg_windows, neg_path, target_fp, max_fp, min_dr, win_size, min_block, step_size)
    level = 0;
    dr = 1.0;
    fp = 1.0;
    n_sample = 250;
    
    cascade_pos_wins = pos_windows;
    cascade_neg_wins = neg_windows;
    
    cascade_svm = {};
    cascade_th = {};
    
    block_set = make_block_set(win_size, min_block);
    
    while fp > target_fp
        level = level+1;
        level_fp = 1.0;
        
        predict = [];
        strong_svm = [];
        
        % pos imgs windows = extract_HOG_for_train & neg windows = extract random images & extract all windows in those images
        % new_neg_windows = find_more_neg_wins(max_sample, ...);

        n_pos = length(cascade_pos_wins);
        n_neg = length(cascade_neg_wins);
        n_win = n_pos + n_neg;
        total_windows = [cascade_pos_wins ; cascade_neg_wins];
        
        weight_pos = ones(n_pos,1).*1/(2*n_pos);
        weight_neg = ones(n_neg,1).*1/(2*n_neg);
        weight = [weight_pos ; weight_neg];
        
        while level_fp > max_fp
            if isfile(sprintf('cascade_svm_%d.mat', level))
                break;
            end
            weight = weight./sum(weight);
            min_eps = -1;

            sample = block_sampling(block_set, n_sample); % list of [x, y, w, h] * n_sample
            
            % 1) train 250 linear SVMs using Pos and Neg samples
            for i=1:n_sample % DEBUG : should transpose? what about weight?
                block_pos = sample(i,:); % selected block = [x, y, w, h]
                
                win_row = block_pos(2):block_pos(2)+block_pos(4)-1;
                win_col = block_pos(1):block_pos(1)+block_pos(3)-1;
                
                sample_blocks = [];
                
                for j=1:n_win
                    sample_blocks = [sample_blocks total_windows{j}(win_row,win_col)];
                end
                sample_hog = extractHOGFeatures(sample_blocks,'CellSize', [block_pos(4)/2 block_pos(3)/2], 'BlockOverlap', [0 0]);
                sample_hog = reshape(sample_hog, 36, n_win);
                sample_hog = sample_hog';
                
                pos_y = ones(n_pos,1);
                neg_y = zeros(n_neg,1);
                sample_y = [pos_y ; neg_y]; 
                
                sample_svm = fitcsvm(sample_hog, sample_y);
                
                sample_dist = sample_hog * sample_svm.Beta + sample_svm.Bias;
                
                % 2) add the best SVM into the strong classifier, update the weight in AdaBoost manner
                for j=1:n_win
                    new_label = sample_dist >= sample_dist(j); % change threshold
                    err = abs(new_label - sample_y);
                    eps = sum(weight.*err);
                    
                    if min_eps == -1
                        min_eps = eps;
                        best_svm = [block_pos (sample_svm.Beta)' (sample_svm.Bias)']; % [x, y, w, h, beta, bias]
                        min_err = err;
                        best_label = new_label;
                        best_th = sample_dist(j);
                    elseif min_eps > eps
                        min_eps = eps;
                        best_svm = [block_pos (sample_svm.Beta)' (sample_svm.Bias)']; % [x, y, w, h, beta, bias]
                        min_err = err;
                        best_label = new_label;
                        best_th = sample_dist(j);
                    end
                end
            end
            
            % score = hog * svm_sample{i}.Beta + svm_sample{i}.Bias
            
            beta = min_eps/(1-min_eps);
            weight = weight.*(beta.^(1-min_err));
            alpha = -log(beta);
            strong_svm = [strong_svm ; best_svm alpha best_th]; % [x, y, w, h, beta, bias, alpha, th]
            
            % 3) evaluate Pos and Neg by current strong clssifier
            predict = [predict best_label.*alpha]; % col = alpha_t * h_t(x) of each round
            total_predict = sum(predict,2); % sum of alpha_t * h_t(x)
            pos_predict = sort(total_predict(1:n_pos), 'descend'); % only sort pos (TP + FN)
            
            % 4) decrease threshold until d_min holds
            th = pos_predict(ceil(min_dr*n_pos));
            strong_label = total_predict >= th;
            
            level_dr = sum(strong_label(1:n_pos)) / n_pos; % Detection rate
            
            % 5) compute f_i(level_fp) under this threshold
            level_fp = sum(strong_label(n_pos+1:end)) / n_neg; % FP rate
        end

        
        if isfile(sprintf('cascade_svm_%d.mat', level))
            load(sprintf('cascade_svm_%d', level),'level', 'cascade_svm', 'cascade_th', 'fp', 'dr');
            
        else
            fp = fp * level_fp;
            dr = dr * level_dr;
            cascade_svm{end+1} = strong_svm;
            cascade_th{end+1} = th;
            cascade_pos_wins = cascade_pos_wins(strong_label(1:n_pos)==1); % TP
            cascade_neg_wins = cascade_neg_wins(strong_label(n_pos+1:end)==1); % FP
        
            save(sprintf('cascade_svm_%d', level), 'level', 'cascade_svm', 'cascade_th', 'fp', 'dr');
            save(sprintf('cascade_pos_wins_%d',level), 'cascade_pos_wins', '-v7.3');
            save(sprintf('cascade_neg_wins_%d',level), 'cascade_neg_wins', '-v7.3');
        end
        
        if isfile(sprintf('cascade_pos_win_%d.mat', level))
            load(sprintf('cascade_pos_wins_%d', level),'cascade_pos_wins');
            load(sprintf('cascade_neg_wins_%d', level),'cascade_neg_wins');
        end
        
        if fp > target_fp
            [hard_example_wins, ~, ~] = extract_multiscale_windows(neg_path, 2, 1.2, [0 0], win_size, step_size); % 2 = rand
            [fp_idx, ~] = detect_cascade_HOG(cascade_svm, cascade_th, hard_example_wins);
            hard_example_wins = hard_example_wins(fp_idx);
            cascade_neg_wins = [cascade_neg_wins ; hard_example_wins];
        end
        fprintf('[train cascade hog] level %d is finished.\n', level);
        % start next cascade level
    end
end