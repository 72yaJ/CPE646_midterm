clear;
close all;
clc;

D1 = [[2;-3;2],[-1;0;-2],[5;-4;2],[-4;4;-3],[-2;3;-2]];
D2 = [[-2;1;4],[1;0;5],[-2;-3;2],[3;-5;7],[1;-1;6]];

% Step1
m1 = mean(D1,2);
m2 = mean(D2,2);

% [row,col] = size(D1);
% S1 = zeros(row);
% for i = 1:col
%     S1 = S1+(D1(:,i)-m1)*(D1(:,i)-m1).';
% end
% [row,col] = size(D2);
% S2 = zeros(row);
% for i = 1:col
%     S2 = S2+(D2(:,i)-m2)*(D2(:,i)-m2).';
% end

[row,col] = size(D1);
x1m = D1-m1*ones(1,col);
[row,col] = size(D2);
x2m = D2-m2*ones(1,col);
S1 = x1m*x1m';
S2 = x2m*x2m';

S_w = S1 + S2;
w = inv(S_w)*(m1-m2);
w = w/norm(w);

y_D1 = w.'*D1;
y_D2 = w.'*D2;

% Step2
syms b
p_x_w1 = 1/(2*b).*exp(-abs(y_D1)./b);
y1 = diff(log(p_x_w1),b);
y = sum(y1);
b = solve(y == 0, b);
b = double(b);

mu2 = mean(y_D2);
sigma2 = sqrt(var(y_D2));

% Step3
syms x
p_x_w1 = 1/(2*b)*exp(-abs(x)/b);
p_x_w2 = normpdf(x,mu2,sigma2);
% figure,
% ezplot(p_x_w1);hold on;
% ezplot(p_x_w2);
% xlim([-6,4]);
% legend('p(x|w_1)','p(x|w_2)');

% Step4
lamda = [0 2;1 0]; % loss function
Pro_w2_w1 = length(y_D2)/length(y_D1); % P(w2)/P(w1)
theta_b = (lamda(1,2)-lamda(2,2))/(lamda(2,1)-lamda(1,1))...
    *Pro_w2_w1; % likelihood ratio threshold
ratio_p_x_w = p_x_w1/p_x_w2; %  likelihood ratio function 
% ??? need to calculate the decision boundary ???
x1 = -10:0.1:10;
l = double(subs(ratio_p_x_w,x,x1));
c1=(l>=theta_b);
c2=(l<theta_b); % verify exceeding the threshold and choose the value
% figure,plot(x1,c1,x1,c2);
% legend('w1','w2');
% ylim([-0.25,1.25]);

% Step5
Pro_w1 = length(y_D1)/(length(y_D1)+length(y_D2)); % P(w1), probability, prior
Pro_w2 = length(y_D2)/(length(y_D1)+length(y_D2)); % P(w2), probability, prior
p_x = p_x_w1*Pro_w1+p_x_w2*Pro_w2; % p(x), evidence
Pro_w1_x = p_x_w1*Pro_w1/p_x; % P(w1|x), probability, posterior
Pro_w2_x = p_x_w2*Pro_w2/p_x;
y_test = w.'*[0;1;1];
Pro_w1_y_test= double(subs(Pro_w1_x,x,y_test));
Pro_w2_y_test= double(subs(Pro_w2_x,x,y_test));
l_y_test = double(subs(ratio_p_x_w,x,y_test));
if l_y_test>=theta_b
    flag = 'w1';
else
    flag = 'w2';
end
R_a1_x = lamda(1,1)*Pro_w1_x+lamda(1,2)*Pro_w2_x; % R(a1|x)
R_a2_x = lamda(2,1)*Pro_w1_x+lamda(2,2)*Pro_w2_x; % R(a2|x)
R_x = vpa(int(R_a1_x*p_x+R_a2_x*p_x,x),3); % should be integral together 
Baysian_y_test = double(subs(R_x,x,y_test));
Baysian_y1_test = double(subs(R_a1_x,x,y_test));
Baysian_y2_test = double(subs(R_a2_x,x,y_test));
r1 = c1.*double(subs(R_a1_x,x,x1));
r2 = c2.*double(subs(R_a2_x,x,x1));
% figure,
% plot(x1,r1,x1,r2);
return





















