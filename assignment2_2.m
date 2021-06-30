%% 2. Weighted Median Filter
clear;
clc;

r = 8;
img = imread("monkey_clean.png");
img1 = imread("monkey_noise1.png");
img2 = imread("monkey_noise2.png");

img_gray = rgb2gray(img);
img1_gray = rgb2gray(img1);
img2_gray = rgb2gray(img2);

[h, w] = size(img1_gray);
filtered = zeros(h,w, 'uint8');

tic;
for i=1:h
    [H, numofel] = make_new_H(img1_gray,i,1,r,1); % Box filter
    m = get_median(H, numofel); % Box filter
    filtered(i,1) = m;
    for j=2:w
        for k=-r/2:r/2
            if(i+k>0 && j-r/2-1>0 && i+k<=256 && j-r/2-1<=256)
                H(img1_gray(i+k,j-r/2-1)+1) = H(img1_gray(i+k,j-r/2-1)+1)-1;
                numofel = numofel-1;
            end
            if(i+k>0 && j+r/2>0 && i+k<=256 && j+r/2<=256)
                H(img1_gray(i+k,j+r/2)+1) = H(img1_gray(i+k,j+r/2)+1)+1;
                numofel = numofel+1;
            end
        end
        m = get_median(H, numofel); % Box filter
        filtered(i,j) = m;
    end
end
toc
peaksnr_1 = psnr(filtered, img_gray);
fprintf('\n The Peak-SNR value of monkey_noise1.png, box filter is %0.4f \n', peaksnr_1);

%figure(1); subplot(1,2,1); imshow(img1_gray); subplot(1,2,2); imshow(filtered);

tic;
for i=1:h
    for j=1:w
        H = zeros(1,256);
        numofel = 0;
        [H, numofel] = make_new_H(img1_gray,i,j,r,2); % Gaussian filter
        m = get_median(H, numofel); % Gaussian filter
        filtered(i,j) = m;
    end
end
toc
peaksnr_1 = psnr(filtered, img_gray);
fprintf('\n The Peak-SNR value of monkey_noise1.png, Gaussian filter is %0.4f \n', peaksnr_1);

%figure(2); subplot(1,2,1); imshow(img1_gray); subplot(1,2,2); imshow(filtered);

tic;
for i=1:h
    for j=1:w
        H = zeros(1,256);
        numofel = 0;
        [H, numofel] = make_new_H(img1_gray,i,j,r,3); % Bilateral filter
        H = H./numofel; % Normalization
        m = get_median(H, 1); % Bilateral filter
        filtered(i,j) = m;
    end
end
toc
peaksnr_1 = psnr(filtered, img_gray);
fprintf('\n The Peak-SNR value of monkey_noise1.png, bilateral filter is %0.4f \n', peaksnr_1);

%figure(3); subplot(1,2,1); imshow(img1_gray); subplot(1,2,2); imshow(filtered);

%% histogram initialize
function [H, numofel] = make_new_H(img, p, q, r, filter)
    
    numofel = 0;
    if (filter == 1) % Box filter
        H = zeros(1,256, 'uint32');
        for i=-r/2:r/2
            for j=-r/2:r/2
                if (p+i>0 && q+j>0 && p+i<=256 && q+j<=256) 
                    H(img(p+i,q+j)+1) = H(img(p+i,q+j)+1) + 1;
                    numofel = numofel + 1;
                end
            end
        end
    end
    if (filter == 2) % Gaussian filter
        H = zeros(1,256);
        for i=-r/2:r/2
            for j=-r/2:r/2
                if (p+i>0 && q+j>0 && p+i<=256 && q+j<=256) 
                    H(img(p+i,q+j)+1) = H(img(p+i,q+j)+1) + gaussian(p,q,p+i,q+j);
                    numofel = numofel + gaussian(p,q,p+i,q+j);
                end
            end
        end
    end
    if (filter == 3) % Bilateral filter
        H = zeros(1,256);
        for i=-r/2:r/2
            for j=-r/2:r/2
                if (p+i>0 && q+j>0 && p+i<=256 && q+j<=256) 
                    H(img(p+i,q+j)+1) = H(img(p+i,q+j)+1) + bilateral(img,p,q,p+i,q+j);
                    numofel = numofel + bilateral(img,p,q,p+i,q+j);
                end
            end
        end
    end
    
end

%% Median
function m = get_median(H, num_of_el)
    count = 0;
    m=0;
    for i=1:256
        count = count + H(i);
        if (count >= num_of_el/2)
            m = i-1;
            break;
        end
    end
end

%% Gaussian filter 
function g = gaussian(p, q, a, b)
    sig = 2;
    g = 1/(2*pi*sig.^2)*exp(-((p-a).^2+(q-b).^2)/sig.^2);
end
%% Bilateral filter
function b = bilateral(img, p, q, a, b)
    sig_s = 2;
    sig_r = 230;
    
    g_s = 1/(2*pi*sig_s.^2)*exp(-((p-a).^2+(q-b).^2)/sig_s.^2);
    g_r = 1/(2*pi*sig_r.^2)*exp(-(double(img(p,q)-img(a,b)).^2)/sig_r.^2);
    b = g_s * g_r;
end