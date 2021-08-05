function [src_X,tar_X, Ws] = mLDJDA(src_X,tar_X,src_label,parameter)

% src_X : n_s*d  instance  of source domain 
% tar_X: n_t*d  instance of  target  domain 
%src_label 1*n_s   label of source domain 
% parameter.noise: corruption level
% parameter.layers: number of layers to stack
% allhx: (layers*d)xn stacked hidden representations
% parameter.lambda: Regularization
% parameter.gamma:trade-off parameter
src_X=src_X';
tar_X=tar_X';
disp('stacking hidden layers...')
xx = [src_X,tar_X];
prevhx = xx;
allhx = [];
Ws={};
% [Y_tar_pseudo]= Pseudolable(src_X, tar_X, src_label); 
          data_src=[src_X',src_label'];
         [pre,dis] = liblinear(data_src,tar_X');
  Y_tar_pseudo=pre';
  %parameter.MMD=eye(size(xx,2));
 parameter.MMD = MMD(src_X, tar_X, src_label,Y_tar_pseudo);
parameter.gaph=Gaph(src_X,tar_X,src_label,Y_tar_pseudo);
%parameter.gaph=eye(size(xx,2));
for layer = 1:parameter.layers
    disp(['layer:',num2str(layer)])
	tic
    [src_X,tar_X, W] = LDJDA(src_X,tar_X,parameter);
    toc
     %[Y_tar_pseudo]= Pseudolable(src_X,tar_X,src_label);
      data_src=[src_X',src_label'];
         [pre,dis] = liblinear(data_src,tar_X');
         Y_tar_pseudo=pre';
     parameter.MMD = MMD(src_X,tar_X, src_label,Y_tar_pseudo);
     parameter.gaph=Gaph(src_X,tar_X,src_label,Y_tar_pseudo);
	Ws{layer} = W;
     newhx=[src_X,tar_X];
	allhx = [allhx; newhx];
end
