function final_SVM = train_HOG(pos_windows, neg_windows, train_neg_path, scale, win_size, step_size)
total_windows = [pos_windows ; neg_windows];
pos_y = ones(length(pos_windows), 1);
neg_y = zeros(length(neg_windows), 1);
y_train = [pos_y ; neg_y];
total_n = length(y_train);
X_train = zeros(total_n, 105*4*9);
parfor i=1:total_n
    X_train(i,:) = extractHOGFeatures(total_windows{i});
end
SVM1 = fitcsvm(X_train, y_train);
final_SVM = train_with_hard_example(SVM1, train_neg_path, X_train, y_train, scale, win_size, step_size);
end