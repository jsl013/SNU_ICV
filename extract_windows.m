function [windows, y] = extract_windows(path, option, padding, win_size, step_size)
    % for train data
    % option: pos(1) or neg(0) of images
    lists = dir(path);
    lists = lists(3:end);
    n_img = length(lists);
    n_sample = 10;
    
    if option == 1
        windows = cell(n_img,1);
        for i=1:n_img
            img_name = strcat(path, lists(i).name);
            img = imread(img_name);

            % delete [16 16] padding (training data)
            img = img(padding(1)+1:end-padding(1),padding(2)+1:end-padding(2),:);
            img = rgb2gray(img);

            windows{i} = img;
        end
        y = ones(n_img, 1); % pos
    elseif option == 0
        windows = cell(n_img*n_sample,1);
        for i=1:n_img
            img_name = strcat(path, '/', lists(i).name);
            img = imread(img_name);
            img = rgb2gray(img);
            
            [row, col] = size(img);
            patch_x = floor((col-win_size(1))/step_size(1)) + 1; % x->col (w)
            patch_y = floor((row-win_size(2))/step_size(2)) + 1; % y->row (h)
            n_patch = patch_y * patch_x;
            
            sample = randsample(n_patch, n_sample); % n_sample from 1~n_patch
            sample = sample - 1; % 0~n_patch-1

            for j=1:n_sample
                sample_row = floor(sample(j)/patch_x);
                sample_col = mod(sample(j),patch_x);
                sample_img = img(1+sample_row*step_size(2):win_size(2)+sample_row*step_size(2),1+sample_col*step_size(1):win_size(1)+sample_col*step_size(1));
                
                windows{(i-1)*n_sample+j} = sample_img;
            end
        end
        y = zeros(n_img*n_sample, 1); % neg
    end
end