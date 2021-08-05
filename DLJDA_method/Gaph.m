function [gaph]=Gaph(src_X,tar_X,src_label,tar_label)
% xx : d*n input 
k=5;
label=[src_label,tar_label];
xx=[src_X,tar_X];
[m,n]=size(xx);
Lr=zeros(n,n);
Lp=zeros(n,n);
Wr=zeros(n,n);
Wp=zeros(n,n);
class=unique(label);
num=0;
for j=1:length(class)
  index_1=find(label==class(j));
  index_2=find(label~=class(j)); 
   Xr=xx(:,index_1);
  Xp=xx(:,index_2);
  X=[Xr,Xp];
  distM=dist(X);  
  nr=length(index_1);
  np=length(index_2);
  distMr=distM(1:nr, 1:nr);
  distMp=distM(nr+1:n,1:nr);
  [valr, ordr]=sort(distMr);
  [valp, ordp]=sort(distMp);
  ordp=ordp+nr*ones(size(ordp));
  
  for i=1:nr
    Wr(ordr(2:k+1,i),i+num)=exp(-(distM(ordr(2:k+1,i),i+num).^2)/2);
    Wp(ordp(2:k+1,i),i+num)=exp(-(distM(ordp(2:k+1,i),i+num).^2)/2);
  end  
  num=nr;
   Dr=diag(sum(Wr));
   Dp=diag(sum(Wp));
  end
   Lr=Dr-Wr;
   Lp=Dp-Wp;
   gaph=Lr-Lp;
 end


