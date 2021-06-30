function lbp_SVM = train_LBP(pos_windows, neg_windows, win_size)
total_windows = [pos_windows ; neg_windows];
pos_y = ones(length(pos_windows),1);
neg_y = zeros(length(neg_windows),1);
y_train = [pos_y ; neg_y];
total_n = length(total_windows);
lbp_feature_length = length(extractLBPFeatures(ones(win_size),'CellSize',[16 16]));
X_lbp = zeros(total_n, lbp_feature_length);
parfor i=1:total_n
    X_lbp(i,:) = extractLBPFeatures(total_windows{i},'CellSize',[16 16]);
end
lbp_SVM = fitcsvm(X_lbp, y_train);
end