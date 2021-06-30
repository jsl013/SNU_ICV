function final_SVM = train_with_hard_example(SVM, train_neg_path, X_train, y_train, scale, win_size, step_size)
  neg = 0;
    
  [fp_windows, ~, ~] = extract_multiscale_windows(train_neg_path, neg, scale, padding, win_size, step_size);
  total_n = legnth(fp_windows);
  fp_hog = zeros(total_n, 105*4*9);
  parfor i=1:total_n
    fp_hog(i,:) = extractHOGFeatures(fp_windows{i});
  end
  
  [label, ~] = predict(SVM, fp_hog);
  fp_hog = fp_hog(label == 1);
  
  X_train = [X_train ; fp_hog];
  y_neg = zeros(size(fp_hog,1), 1);
  y_train = [y_train ; y_neg];

  final_SVM = fitcsvm(X_train, y_train);
end
