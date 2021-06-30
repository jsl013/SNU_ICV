%% Homography & Image Stitching
clear;
clc;

% Feature Extraction
img_path = './img_5_3/';
n = 5; % # of imgs : 3 or 5
img_size = 256;
img_points = 300; % 200~300 points in each image

imgs = read_imgs(img_path, n, img_size); % I1, I2, I3
grey_imgs = rgbs2greys(imgs, n);

[F, D] = get_sift(grey_imgs, n, img_points);

% Feature Matching
matches = match_d(D, n);

% Homography Estimation using RANSAC
[H, matches] = HbyRANSAC(F, matches, n);
[DLT_H, DLT_matches] = DLT_optimization(F, H, matches, n);
[LM_H, LM_matches] = LM_optimization(F, H, matches, n);

% Plot Feature Matches
plot_matches(imgs, F, matches, n);
plot_matches(imgs, F, DLT_matches, n);
plot_matches(imgs, F, LM_matches, n);

% Warping Images
panorama_img = image_stitch(imgs, H, n);
figure(); imshow(panorama_img);
panorama_DLT = image_stitch(imgs, DLT_H, n);
figure(); imshow(panorama_DLT);
panorama_LM = image_stitch(imgs, LM_H, n);
figure(); imshow(panorama_LM);

%% Functions for Feature Extraction
function imgs = read_imgs(img_path, n, resize_n)
    if n == 3
        files = {'1.jpg', '2.jpg', '3.jpg'};
    end
    if n == 5
        files = {'1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg'};
    end
    imgs = cell(n,1);
    
    for i=1:n
        imgs{i} = imresize(imread(strcat(img_path, files{i})), [resize_n, resize_n]);
    end
end

function grey_imgs = rgbs2greys(imgs, n)
    grey_imgs = cell(n, 1);

    for i=1:n
        grey_imgs{i} = single(rgb2gray(imgs{i}));
    end
end

function [F, D] = get_sift(imgs, n, img_points)
    F = cell(n, 1);
    D = cell(n ,1);
    
    for i=1:n
        [F{i}, D{i}] = vl_sift(imgs{i}, 'FirstOctave', -1); 
        [F{i}, D{i}] = get_unique(F{i}, D{i}); % avoid multiple detections at the same point
        [F{i}, D{i}] = NMS(F{i}, D{i}, img_points);
    end
end

function [F_unique, D_unique] = get_unique(F, D)
    [~, idx] = sort(F(1,:));
    F_sorted = F(:,idx);
    D_sorted = D(:,idx);

    unique_idx = 1;

    for i = 2:size(F, 2)
        if (F_sorted(1, i) ~= F_sorted(1, unique_idx(end)) || F_sorted(2, i) ~= F_sorted(2, unique_idx(end)))
            unique_idx = [unique_idx, i];
        end
    end
    
    F_unique = F_sorted(:, unique_idx);
    D_unique = D_sorted(:, unique_idx);
end

% Extract features evenly across an image 
function [F_extracted, D_extracted] = NMS(F, D, img_points)
    candidates = zeros(1, size(F,2));
    n = size(F, 2);
    th = 0.8;
    count = 0;
    
    for i=1:n % candidate = ith point of F
        min_d = inf; % min. distance
        for j=1:n % comparison target = jth point of F
            if i == j
                continue
            else
                if F(3,i)/F(3,j) < th % score
                    dist = (F(1,i) - F(1,j)).^2 + (F(2,i) - F(2,j)).^2;
                    if dist < min_d % same
                        min_d = dist;
                    end
                end
            end
        end
        if min_d ~= inf
            count = count + 1;
        end
        candidates(i) = min_d;
    end
    
    fprintf('# of extracted points by NMS: %d\n', count); % for debugging
    
    [~, elected_idx] = sort(candidates, 'descend');
    elected_idx = elected_idx(1:img_points); % choose #(img_points) points of F that the min. distance to others is large 
    F_extracted = F(:, elected_idx);
    D_extracted = D(:, elected_idx);
end

function matches = match_d(D, n)
    matches = cell(n-1,1);
    th = 0.8;
    for i=1:n-1
        matches{i} = vl_ubcmatch(D{i}, D{i+1}, th);
    end
end

%% Functions for Homography Estimation using RANSAC
function [H, matches] = HbyRANSAC(F, matches, n)
    s = 4;
    p = 0.99;
    t = 1.25;
    n_sample = 4;
    H = cell(n-1,1);
    
    for i=1:n-1
        N = inf;
        sample_count = 0;
        max_n_inliers = -1;
        
        F1 = F{i};
        F2 = F{i+1};
        match = matches{i};
        
        x_img1 = F1(1:2,match(1,:));
        x_img2 = F2(1:2,match(2,:)); 
        n_match = size(match,2);

        while N > sample_count
            rand = randsample(n_match, n_sample);
            
            x1 = x_img1(:,rand);
            x2 = x_img2(:,rand);
            
            h = DLT(x1, x2);
            
            inliers = find_inliers(x_img1, x_img2, h, t);
            n_inliers = size(inliers, 2);
            
            if n_inliers > max_n_inliers
                max_n_inliers = n_inliers;
                max_inliers = inliers;
                H{i} = h;
            end
            
            eps = 1 - n_inliers/n_match;
            
            N = abs(log(1-p)/log(1-(1-eps)^s));
            sample_count = sample_count + 1;
        end
        match = matches{i};
        matches{i} = match(:,max_inliers);
        
        fprintf('n_inliers of %d-%d: %d\n', i, i+1, max_n_inliers');

        x1 = [x_img1 ; ones(1,size(x_img1,2))];
        x2 = [x_img2 ; ones(1,size(x_img2,2))];
        
        Hx1 = h*x1;
        H_invx2 = h\x2;
        Hx1 = Hx1./Hx1(3,:);
        H_invx2 = H_invx2./H_invx2(3,:);
        
        d = 0;
        for j=1:size(x_img1,2)
            d = d + (x2(1,j)-Hx1(1,j)).^2 + (x2(2,j)-Hx1(2,j)).^2 + (x1(1,j)-H_invx2(1,j)).^2 + (x1(2,j)-H_invx2(2,j)).^2;
        end
        
        fprintf("symmetric transfer error for img %d, %d : %f\n", i, i+1, d);
    end
end

function H = DLT(x1, x2)
    n_sample = size(x1,2);
    A = zeros(2*n_sample, 9);
    for i=1:n_sample
        A(2*i-1,:) = [0 0 0 x1(:,i)' 1 -x2(2,i)*x1(:,i)' -x2(2,i)];
        A(2*i,:) = [x1(:,i)' 1 0 0 0 -x2(1,i)*x1(:,i)' -x2(1,i)];
    end
    [~, ~, V] = svd(A, 0);
    H = reshape(V(:, end), [3, 3])';
end

function inliers = find_inliers(x1, x2, H, t)
    x1 = [x1 ; ones(1,size(x1,2))];
    x2 = [x2 ; ones(1,size(x2,2))];
    Hx1 = H*x1;
    H_invx2 = H\x2;
    Hx1 = Hx1./Hx1(3,:);
    H_invx2 = H_invx2./H_invx2(3,:);
    
    inliers = [];
    
    for i=1:size(x1,2)
        d = (x2(1,i)-Hx1(1,i)).^2 + (x2(2,i)-Hx1(2,i)).^2 + (x1(1,i)-H_invx2(1,i)).^2 + (x1(2,i)-H_invx2(2,i)).^2;
        if d < t
            inliers = [inliers i];
        end
    end
end
%% Functions for Optimization
function [H, new_matches] = DLT_optimization(F, H, matches, n)
    new_matches = cell(n-1,1);
    
    for i=1:n-1
        match = matches{i};
        F1 = F{i};
        F2 = F{i+1};
        while 1
            x1 = F1(1:2, match(1,:));
            x2 = F2(1:2, match(2,:));
            H{i} = DLT(x1, x2);
            added_match = guided_matching(H{i}, F1, F2, match);
            if size(added_match,1) == 0 && size(added_match,2) == 0
                break;
            else
                match = [match added_matches];
            end
        end
        new_matches{i} = match;
    end
end

function added_match = guided_matching(H, F1, F2, match)
    t = 1.25;
    candidate_x1s = [];
    candidate_x2s = [];
    
    % find points that are not in match
    for i=1:size(F1,2)
        if ~find(match(1,:)==i)
            candidate_x1s = [candidate_x1s i];
        end
    end
    for i=1:size(F2,2)
        if ~find(match(2,:)==i)
            candidate_x2s = [candidate_x2s i];
        end
    end
    
    % find matching points that satisfy DLT condition
    x1s = F1(1:2,candidate_x1s);
    x2s = F2(1:2,candidate_x2s);
    added_match = [];
    
    for i=1:size(x1s,2)
        min_dist = -1;
        min_j = -1;
        x1 = [x1s(:,i) ; 1];
        for j=1:size(x2s,2)
            x2 = [x2s(:,j) ; 1];
            Hx1 = H*x1;
            H_invx2 = H\x2;
            Hx1 = Hx1./Hx1(3);
            H_invx2 = H_invx2./H_invx2(3);
            d = (x2(1)-Hx1(1)).^2 + (x2(2)-Hx1(2)).^2 + (x1(1)-H_invx2(1)).^2 + (x1(2)-H_invx2(2)).^2;
            if min_dist == -1
                if d < t
                    min_d = d;
                    min_j = j;
                end
            else
                if d < t && d < min_d
                    min_d = d;
                    min_j = j;
                end
            end
        end
        min_match = [candidate_x1s(i); candidate_x2s(min_j)];
        added_match = [added_match min_match];
    end
end

function [H, new_matches] = LM_optimization(F, H, matches, n) 
    new_matches = cell(n-1,1);
    lb = [];
    ub = [];
    for i=1:n-1
        F1 = F{i};
        F2 = F{i+1};
        match = matches{i};
        while 1
            x1s = [F1(1:2, match(1,:)) ; ones(1, size(match, 2))];
            x2s = [F2(1:2, match(2,:)) ; ones(1, size(match, 2))];
            x12 = [x1s ; x2s]; % x1: row=1~3, x2: row4~6
            y = zeros(1, size(match,2));
            options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt','Display','off');
            H{i} = lsqcurvefit(@SSE, H{i}, x12, y, lb, ub, options);
            
            added_match = guided_matching(H{i}, F1, F2, match);
            if size(added_match,1) == 0 && size(added_match,2) == 0
                break;
            else
                match = [match added_matches];
            end
        end
        new_matches{i} = match;
    end
end

function err = SSE(H, x)
    n_points = size(x,2);
    err = zeros(1,n_points);
    for i=1:n_points
        x1 = x(1:3,i);
        x2 = x(4:6,i);
        Hx1 = H*x1;
        H_invx2 = H\x2;
        Hx1 = Hx1./Hx1(3);
        H_invx2 = H_invx2./H_invx2(3);
        err(i) = (x2(1)-Hx1(1)).^2 + (x2(2)-Hx1(2)).^2 + (x1(1)-H_invx2(1)).^2 + (x1(2)-H_invx2(2)).^2;
    end
end
%% Functions for Warping Images
function panorama_img = image_stitch(imgs, H, n)
    img_size = size(imgs{1}, 1);
    center = ceil(n/2); % assume n is an odd number
    H_center = cell(n,1);
    tforms = cell(n,1);
    warped_imgs = cell(n,1);
    masks = cell(n,1);
    
    H_center{center} = eye(3); % 3x3 I matrix
    H_down = eye(3);
    H_up = eye(3);
    
    for i=1:center-1
        H_down = H_down*H{center-i};
        H_up = H_up/H{center+i-1};
        H_center{center-i} = H_down;
        H_center{center+i} = H_up;
    end
    for i=1:n
        tforms{i} = projective2d(transpose(H_center{i})); % same as maketform
    end
    
    % assume that (1,1), (1,img_size) of leftmost_image & (img_size,1),
    % (img_size,img_size) of rightmost image decide the size of panorama image
    
    corner1 = H_center{1}*[1 1 1]';
    corner2 = H_center{1}*[1 img_size 1]';
    corner3 = H_center{n}*[img_size 1 1]';
    corner4 = H_center{n}*[img_size img_size 1]';
    corner1 = corner1./corner1(3);
    corner2 = corner2./corner2(3);
    corner3 = corner3./corner3(3);
    corner4 = corner4./corner4(3);
    
    x_min = min([corner1(1) corner2(1) corner3(1) corner4(1)]);
    x_max = max([corner1(1) corner2(1) corner3(1) corner4(1)]);
    y_min = min([corner1(2) corner2(2) corner3(2) corner4(2)]);
    y_max = max([corner1(2) corner2(2) corner3(2) corner4(2)]);
    
    h = ceil(y_max - y_min);
    w = ceil(x_max - x_min);
    
    panorama_img = zeros(h, w, 3, 'uint8');
    
    panorama_ref = imref2d([h, w], [x_min, x_max], [y_min, y_max]);
    
    for i = 1:n
        warped_imgs{i} = imwarp(imgs{i}, tforms{i}, 'OutputView', panorama_ref);
        masks{i} = imwarp(true(size(imgs{i})), tforms{i}, 'OutputView', panorama_ref);
        panorama_img = uint8(~masks{i}).*uint8(panorama_img) + uint8(warped_imgs{i}); % if two images overlap, use pixels of the next image
    end
end

%% Plot images
function plot_matches(imgs, F, matches, n)
    for i=1:n-1
        F1 = F{i};
        F2 = F{i+1};
        img1 = imgs{i};
        img2 = imgs{i+1};
        match = matches{i};
        
        figure();
        imshow([img1, img2]);
        
        xa = F1(1,match(1,:)) ;
        xb = F2(1,match(2,:)) + size(img1,2) ;
        ya = F1(2,match(1,:)) ;
        yb = F2(2,match(2,:)) ;

        hold on ;
        
        line([xa ; xb], [ya ; yb], 'color', 'r') ;
        for j = 1:size(F1,2)
            plot(F1(1,j), F1(2,j), '-o', 'color', 'b');
        end
        for j = 1:size(F2,2)
            plot(F2(1,j)+size(img1,2), F2(2,j), '-o', 'color', 'b');
        end

        drawnow();
        hold off;
    end
end
