%% 1. Weighted Guided Filter
clear;
clc;

img = double(imread("afghan_clean.png"))/255;
img1 = double(imread("afghan_noise1.png"))/255;
img2 = double(imread("afghan_noise2.png"))/255;

[h, w, c] = size(img1);
%% 1-1. grey image
img1_grey = (img1(:,:,1)+img1(:,:,2)+img1(:,:,3))/3;
img2_grey = (img2(:,:,1)+img2(:,:,2)+img2(:,:,3))/3;

I1 = img1_grey; I2 = img2_grey; p1 = I1; p2 = I2;

q1 = guidedfilter(p1, I1, r, eps);
q2 = guidedfilter(p2, I2, r, eps);

figure(1); subplot(1,2,1); imshow(I1); subplot(1,2,2); imshow(q1);
figure(2); subplot(1,2,1); imshow(I2); subplot(1,2,2); imshow(q2);

%% 1-2. use one of RGB color
img1_R = img1(:,:,1); img1_G = img1(:,:,2); img1_B = img1(:,:,3);
I_R = img1_R; p_R = I_R; I_G = img1_G; p_G = I_G; I_B = img1_B; p_B = I_B;

q_R = guidedfilter(p_R, I_R, r, eps); q_G = guidedfilter(p_G, I_G, r, eps); q_B = guidedfilter(p_B, I_B, r, eps);
I = zeros(h, w, c, 'uint8'); q = zeros(h, w, c, 'uint8'); 
I(:,:,1) = uint8(I_R*255);
q(:,:,1) = uint8(q_R*255);
figure(3); subplot(1,2,1); imshow(I); subplot(1,2,2); imshow(q);

I = zeros(h, w, c, 'uint8'); q = zeros(h, w, c, 'uint8'); 
I(:,:,2) = uint8(I_G*255);
q(:,:,2) = uint8(q_G*255);
figure(4); subplot(1,2,1); imshow(I); subplot(1,2,2); imshow(q);

I = zeros(h, w, c, 'uint8'); q = zeros(h, w, c, 'uint8'); 
I(:,:,3) = uint8(I_B*255);
q(:,:,3) = uint8(q_B*255);
figure(5); subplot(1,2,1); imshow(I); subplot(1,2,2); imshow(q);

img2_R = img2(:,:,1); img2_G = img2(:,:,2); img2_B = img2(:,:,3);
I_R = img2_R; p_R = I_R; I_G = img2_G; p_G = I_G; I_B = img2_B; p_B = I_B;

q_R = guidedfilter(p_R, I_R, r, eps); q_G = guidedfilter(p_G, I_G, r, eps); q_B = guidedfilter(p_B, I_B, r, eps);
I = zeros(h, w, c, 'uint8'); q = zeros(h, w, c, 'uint8'); 
I(:,:,1) = uint8(I_R*255);
q(:,:,1) = uint8(q_R*255);
figure(6); subplot(1,2,1); imshow(I); subplot(1,2,2); imshow(q);

I = zeros(h, w, c, 'uint8'); q = zeros(h, w, c, 'uint8'); 
I(:,:,2) = uint8(I_G*255);
q(:,:,2) = uint8(q_G*255);
figure(7); subplot(1,2,1); imshow(I); subplot(1,2,2); imshow(q);

I = zeros(h, w, c, 'uint8'); q = zeros(h, w, c, 'uint8'); 
I(:,:,3) = uint8(I_B*255);
q(:,:,3) = uint8(q_B*255);
figure(8); subplot(1,2,1); imshow(I); subplot(1,2,2); imshow(q);
% Result: 크게 다른 결과는 없었다

%% 1-3. Multi-channel input: RGB color image
clc;

r = 4; eps = 0.2.^2;

I=img;
I_RGB = img1;
p_RGB = I_RGB;
tic;
q_RGB = guidedfilter(p_RGB, I_RGB, r, eps);
toc
peaksnr_1 = psnr(q_RGB, I);
fprintf('\n The Peak-SNR value of afghan_noise1.png is %0.4f \n', peaksnr_1);
%figure(9); subplot(1,2,1); imshow(I_RGB); subplot(1,2,2); imshow(q_RGB);

I_RGB = img2;
P_RGB = I_RGB;
tic;
q_RGB = guidedfilter(p_RGB, I_RGB, r, eps);
toc
peaksnr_2 = psnr(q_RGB, I);
fprintf('\n The Peak-SNR value of afghan_noise2.png is %0.4f \n', peaksnr_2);
%figure(10); subplot(1,2,1); imshow(I_RGB); subplot(1,2,2); imshow(q_RGB);

%% Guided filter function
function q = guidedfilter(p, I, r, eps)
    mean_I = gaussfilter(I, r);
    mean_p = gaussfilter(p, r);
    corr_I = gaussfilter(I.*I, r);
    corr_Ip = gaussfilter(I.*p, r);
    
    var_I = corr_I - mean_I.*mean_I;
    cov_Ip = corr_Ip - mean_I.*mean_p;
    
    a = cov_Ip./(var_I + eps);
    b = mean_p - a.*mean_I;
    
    mean_a = gaussfilter(a, r);
    mean_b = gaussfilter(b, r);
    
    q = mean_a.*I + mean_b;
end
%% f_mean of guassian filter
function filtered = gaussfilter(img, r)
    sig = 2;
    
    is_2D = 1;
    if (numel(size(img)) == 3)
        [h, w, c] = size(img);
        is_2D = 0;
    else
        [h, w] = size(img);
    end
    
    %assume r is odd number
    filter = zeros(r, r);
    for i=1:r 
        for j=1:r
            filter(i,j) = 1/(2*pi*sig.^2)*exp(-((i-(fix(r/2)+1)).^2+(j-(fix(r/2)+1)).^2)/sig.^2);
        end
    end

    if (is_2D) 
        filtered = conv2(img, filter, 'same');
        norm = conv2(ones(h,w), filter, 'same');
    else
        filtered = convn(img, filter, 'same');
        norm = convn(ones(h,w,c), filter, 'same');
    end
    filtered = filtered./norm;
end