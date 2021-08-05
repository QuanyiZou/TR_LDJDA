function [new_src_X,new_tar_X, W] = LDJDA(src_X,tar_X,parameter)
% xx : dxn input
% noise: corruption level
% lambda: regularization
 % hx: dxn hidden representation
% W: dx(d+1) mapping
n_s=size(src_X,2);
n_t=size(tar_X,2);
xx = [src_X,tar_X];
[d, n] = size(xx);
% adding bias
xxb = [xx; ones(1, n)];

% scatter matrix S
S = xxb*xxb'; 

% corruption vector
q = ones(d+1, 1)*(1-parameter.noises);
q(end) = 1;


% Q: (d+1)x(d+1)
Q = S.*(q*q');
Q(1:d+2:end) = q.*diag(S); %dag ¶Ô½ÇÏß

% P: dx(d+1)
P = S(1:end-1,:).*repmat(q', d, 1);

MMD = xxb*(parameter.MMD+parameter.gamma*parameter.gaph)*xxb';
MMD = MMD.*(q*q');
MMD(1:d+2:end) = q.*diag(MMD);
reg = parameter.lambda*eye(d+1);
% A = diag(S);
% for i=1:size(Q,1)
%     reg(i,i) = lambda*A(i);
% end
% % reg = lambda*diag(S);
reg(end,end) = 0;
W = P/(Q+reg+MMD);
% W = P/(Q+reg +beta*MMD +parameter.gamma*Manifold);

hx = W*xxb;
hx = tanh(hx);
new_src_X=hx(:,1:n_s);
new_tar_X=hx(:,n_s+1:end);
% hx = sigm(hx);