%% Sliding Window-based Object Detection
% implement human detection algorithm using HOG & SVM
% 1) make HOG(extractHOGFeatures(I)) vs. cacade of HOG -> compare!!
% 2) train linear SVM (dataset : INRIA person dataset)
% 3) implement overall detection pipeline using classifier (2) 
%   - single scale detector -> full multi-scale detector
%   - use non-maximum suppression step : rm duplicated detections

% test & evaluate the final classifier
% 4) test
% 5) evaluate performance: ROC curve, precision-recall curve, avg precision
%   - use evaluate_detections.m
% 6) visualize the detection result
%   - use visualize_detections_by_image.m or
%   visualize_detections_by_image_no_gt.m

% Extra Optimization
% 7) extra point : !!! use local binary pattern (LBP) feature instead of HOG !!! (extractLBPFeatures)
% 8) Optimization: How to improve the result?

% parameters
pos = 1;
neg = 0;
step_size = [8 8];
scale = 1.2;
win_size = [64 128];
min_size = [64 128];
min_block = [12 12];
train_padding = [16 16];
test_padding = [0 0];

target_fp = 0.01;
max_fp = 0.7;
min_dr = 0.9975;

train_pos_path = './dataset/INRIAPerson/train_96x160/pos/';
train_neg_path = './dataset/INRIAPerson/train_96x160/neg/';
test_pos_path = './dataset/INRIAPerson/Test/pos/';
test_neg_path = './dataset/INRIAPerson/Test/neg/';
label_path = 'label.txt';
%% Prepare Train data
% Extract windows from train data
[pos_windows, pos_y] = extract_windows(train_pos_path, pos, train_padding, win_size, step_size);
[neg_windows, neg_y] = extract_windows(train_neg_path, neg, [0 0], win_size, step_size); 
%% HOG with SVM
% Train linear SVM using fitcsvm
final_SVM = train_HOG(pos_windows, neg_windows, train_neg_path, scale, win_size, step_size);
% Train linear SVM
test_HOG(final_SVM, test_pos_path, scale, win_size, step_size, label_path);
%% Cascade of HOG with SVM
% Train cascade of HOG
[cascade_svm, cascade_th] = train_cascade_HOG(pos_windows, neg_windows, test_neg_path, target_fp, max_fp, min_dr, win_size, min_block, step_size);
% Test cascade of HOG
test_cascade_HOG(cascade_svm, cascade_th, test_pos_path, scale, win_size, step_size, label_path);
%% LBP with SVM
% Train SVM using LBP feature
lbp_SVM = train_LBP(pos_windows, neg_windows, win_size);
% Test SVM of LBP
test_LBP(lbp_SVM, test_pos_path, scale, win_size, step_size, label_path);