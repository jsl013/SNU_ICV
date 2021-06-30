%% 2D Fourier Transform of Images
clear;
clc;

img1 = imread("img1.jpg");
img2 = imread("img2.jpg");

figure(1);
subplot(1,2,1); imshow(img1); 
subplot(1,2,2); imshow(img2);

resizedCheckerboard = imresize(img2, size(img1));

img1_gray = mat2gray(img1);
img2_gray= mat2gray(resizedCheckerboard);

f1=fft2(img1_gray); f2=fft2(img2_gray);
f1=fftshift(f1); f2=fftshift(f2);

img1_mag=abs(f1); img2_mag=abs(f2);
img1_phase=angle(f1); img2_phase=angle(f2);

figure(2);
subplot(1,2,1); imshow(log(img1_mag)); title("img1 mag"); 
subplot(1,2,2); imshow(log(img2_mag)); title("img2 mag");

figure(3);
subplot(1,2,1); imshow(img1_phase); title("img1 phase"); 
subplot(1,2,2); imshow(img2_phase); title("img2 phase");

new_f1=img1_mag.*exp(1j*img2_phase);
new_img1=ifft2(ifftshift(new_f1));

new_f2=img2_mag.*exp(1j*img1_phase);
new_img2=ifft2(ifftshift(new_f2));

figure(4);
subplot(1,2,1); imshow(real(new_img1)); title("img1 mag + img2 phase");
subplot(1,2,2); imshow(real(new_img2)); title("img2 mag + img1 phase");

%% Perspective Image Transforms

M1=[1.6322 0 0; 0.2120 1.6336 0.0013; -101.9757 -0.6322 1]';
M2=[1.4219 0.3183 0.0013; 0 1.4206 0; -0.4206 -101.8704 1]';
M3=[0.7033 -0.2239 -0.0009; 0 0.9991 0; 0.2958 0.2239 1]';
M4=[1.1044 -0.3493 0.0003; 0.0011 1.5066 0.0011; -0.1041 -0.1560 1]';

projected1=zeros(375,500);
projected2=zeros(375,500);
projected3=zeros(375,500);
projected4=zeros(375,500);

% forward warping - holes

for i=1:375
    for j=1:500
        tmp=M1*[i j 1]';
        u=round(tmp(1,1)/tmp(3,1));
        v=round(tmp(2,1)/tmp(3,1));
        if(u>0 && v>0 && u<376 && v<501)
            projected1(u,v)=img1_gray(i,j);
        end
    end
end

for i=1:375
    for j=1:500
        tmp=M2*[i j 1]';
        u=round(tmp(1,1)/tmp(3,1));
        v=round(tmp(2,1)/tmp(3,1));
        if(u>0 && v>0 && u<376 && v<501)
            projected2(u,v)=img1_gray(i,j);
        end
    end
end

for i=1:375
    for j=1:500
        tmp=M3*[i j 1]';
        u=round(tmp(1,1)/tmp(3,1));
        v=round(tmp(2,1)/tmp(3,1));
        if(u>0 && v>0 && u<376 && v<501)
            projected3(u,v)=img1_gray(i,j);
        end
    end
end

for i=1:375
    for j=1:500
        tmp=M4*[i j 1]';
        u=round(tmp(1,1)/tmp(3,1));
        v=round(tmp(2,1)/tmp(3,1));
        if(u>0 && v>0 && u<376 && v<501)
            projected4(u,v)=img1_gray(i,j);
        end
    end
end

% forward warping
figure(5);
subplot(1,4,1); imshow(projected1);
subplot(1,4,2); imshow(projected2);
subplot(1,4,3); imshow(projected3);
subplot(1,4,4); imshow(projected4);


% backward warping - no holes

for i=1:375
    for j=1:500
        tmp=M1\[i j 1]';
        u=round(tmp(1,1)/tmp(3,1));
        v=round(tmp(2,1)/tmp(3,1));
        if(u>0 && v>0 && u<376 && v<501)
            projected1(i,j)=img1_gray(u,v);
        end
    end
end

for i=1:375
    for j=1:500
        tmp=M2\[i j 1]';
        u=round(tmp(1,1)/tmp(3,1));
        v=round(tmp(2,1)/tmp(3,1));
        if(u>0 && v>0 && u<376 && v<501)
            projected2(i,j)=img1_gray(u,v);
        end
    end
end

for i=1:375
    for j=1:500
        tmp=M3\[i j 1]';
        u=round(tmp(1,1)/tmp(3,1));
        v=round(tmp(2,1)/tmp(3,1));
        if(u>0 && v>0 && u<376 && v<501)
            projected3(i,j)=img1_gray(u,v);
        end
    end
end

for i=1:375
    for j=1:500
        tmp=M4\[i j 1]';
        u=round(tmp(1,1)/tmp(3,1));
        v=round(tmp(2,1)/tmp(3,1));
        if(u>0 && v>0 && u<376 && v<501)
            projected4(i,j)=img1_gray(u,v);
        end
    end
end

% backward warping
figure(6);
subplot(1,4,1); imshow(projected1);
subplot(1,4,2); imshow(projected2);
subplot(1,4,3); imshow(projected3);
subplot(1,4,4); imshow(projected4);

imwrite(projected1, "projected1.png");
imwrite(projected2, "projected2.png");
imwrite(projected3, "projected3.png");
imwrite(projected4, "projected4.png");

%% Camera Calibration - (b)
clear;
clc;

image_points=[880 214; 
               43 203;
               270 197;
               886 347;
               745 302;
               943 128;
               476 590;
               419 214;
               317 335;
               783 521;
               235 427;
               665 429;
               655 362;
               427 333;
               412 415;
               746 351;
               434 415;
               525 234;
               716 308;
               602 187];
 
world_points =[312.747 309.140 30.086;
               305.796 311.649 30.356;
               307.694 312.358 30.418;
               310.149 307.186 29.298;
               311.937 310.105 29.216;
               311.202 307.572 30.682;
               307.106 306.876 28.660;
               309.317 312.490 30.230;
               307.435 310.151 29.318;
               308.253 306.300 28.881;
               306.650 309.301 28.905;
               308.069 306.831 29.189;
               309.671 308.834 29.029;
               308.255 309.955 29.267;
               307.546 308.613 28.963;
               311.036 309.206 28.913;
               307.518 308.175 29.069;
               309.950 311.262 29.990;
               312.160 310.772 29.080;
               311.988 312.709 30.514];

N=20;
A=zeros(2*N,12);
zero_vector=[0 0 0 0];
           
for i=1:20
    u=image_points(i,1);
    v=image_points(i,2);
    X=world_points(i,:);
    A(2*i-1,:)=[X 1 zero_vector -u*X -u];
    A(2*i,:)=[zero_vector X 1 -v*X -v];
end

% find projection matrix P by using the SVD method
[U, S, V] = svd(A);
P1=V(:,end);
P1=reshape(P1,[],3)';
P1=P1/P1(3,4) % normalize to compare with the result of pseudo inverse method


%% Camera Calibration - (c)

N=20;
A=zeros(2*N,11);
b=zeros(2*N,1);
zero_vector=[0 0 0 0];

for i=1:20
    u=image_points(i,1);
    v=image_points(i,2);
    X=world_points(i,:);
    A(2*i-1,:)=[X 1 zero_vector -u*X];
    A(2*i,:)=[zero_vector X 1 -v*X];
    b(2*i-1,1)=u;
    b(2*i,1)=v;
end

% find projection matrix P by using a pseudo inverse method
A_plus=(A'*A)\A';
P2=A_plus*b;
P2=[P2; 1]; % assume m_34=1
P2=reshape(P2,[],3)'

