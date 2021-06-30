function [total_patch, position, n_img] = extract_multiscale_windows(path, option, scale, padding, win_size, step_size)
    % for test data
    % option: pos(1) or neg(0) or rand(2) of images
    % win_size = [w h]
    % step_size = [s_x s_y]
    n_test_img = 5;
    
    % init outputs
    position = [];
    total_patch = [];
    
    lists = dir(path);
    lists = lists(3:end);
    n_img = length(lists);
    
    if option == 1 || option == 0
        img_num = 1:n_img;
    else
        img_num = randsample(n_img, n_test_img);
    end
    
    for iter=1:length(img_num)
        i = img_num(iter);
        % for each image, make HOG features for each layer
        img_name = strcat(path, '/', lists(i).name);
        img = imread(img_name);
        
        % delete padding
        img = img(padding(1)+1:end-padding(1),padding(2)+1:end-padding(2),:);
        img = rgb2gray(img);
        [row, col] = size(img);
        layer = 0;
        
        while row >= 128 && col >= 64
            % 1) image pyramid
            img = imresize(img, [row col]);
            patch_x = floor((col-win_size(1))/step_size(1)) + 1; % x->col (w)
            patch_y = floor((row-win_size(2))/step_size(2)) + 1; % y->row (h)
            n_patch = patch_y * patch_x;
        
            % patch_HOG & patch_position is HOG & position of patches from one image
            %patch_HOG = zeros(n_patch, n_bin*block_size*n_block); % should check!!
            patch_position = zeros(n_patch, 6); % it contains pos/neg, img#, layer#, lefttop_pos, width/height (layer#=0=origimg)
            patch = cell(n_patch,1);
            %patch_cnt = 1;

            % 2) extract patches & extract postions, HOG feature of the patches from one image
            for k=0:patch_x-1
                for l=0:patch_y-1
                    lefttop = [1+k*step_size(1) 1+l*step_size(2)]; % [x, y]
                    wh = [win_size(1) win_size(2)]; % [w, h]
                    patch{k*patch_y+l+1} = img(lefttop(2):lefttop(2)+wh(2)-1, lefttop(1):lefttop(1)+wh(1)-1); % x->col, y->row

                    %patch_HOG(k*patch_y+l+1,:) = extractHOGFeatures(patch{k*patch_y+l+1});
                    % patch_position contains pos/neg, img#, lefttop_pos(x,y), width/height
                    patch_position(k*patch_y+l+1,:) = [option i floor((lefttop-1).*scale.^layer)+1 floor(wh.*scale.^layer)];
                    %patch_cnt = patch_cnt + 1;
                end
            end
            
            position = [position ; patch_position];
            total_patch = [total_patch ; patch];

            row = floor(row/scale);
            col = floor(col/scale);
            layer = layer + 1;
        end
    end
end
