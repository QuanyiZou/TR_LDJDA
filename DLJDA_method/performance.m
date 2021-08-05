function [pd,pf,f_measure,g_measure,balance,MCC,Auc] = performance(Y_pesudo, Y_ture,dis)
% input
    %%% Y_ture： a column vetor, each row is an instance's class label {-1,1}, -1 denotes nondefective, 1 denotes defective.
    %   %%% Y_pesudo： prediction label， which has the same size as actual_label.
    %%%%  dis  probability of prediction label
% output: 
    %%%  evaluation indicators. 
[A,~] = confusionmat(Y_ture, Y_pesudo); % confusion function
tp=A(2,2); 
fn=A(2,1);
tn=A(1,1);
fp=A(1,2);
precision = tp/(tp +fp);
pd=tp/(tp+ fn); 
specificity=A(1,1)/(A(1,1)+A(1,2));
%gmean=sqrt(specificity*recall);
g_measure=2*pd*specificity/(specificity+pd);
b=2;
recall=pd;
pf=1-specificity;
f_measure=((1+b^2) * precision * recall)./(b^2*precision + recall);
MCC = (tp*tn - fp*fn)./ sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));
balance=1-(sqrt((1-pd)^2+(0-pf)^2)/sqrt(2));
Auc=AUC(Y_ture ,dis);
if isnan(f_measure)
    f_measure = 0;
end
    
if isnan(MCC)
    MCC = 0;
end

  if isnan(Auc)
    Auc = 0;
  end
end


function [result]=AUC(test_targets,output) 
%计算AUC值,test_targets为原始样本标签,output为分类器得到的判为正类的概率
[B,I]=sort(output); 
M=0;N=0; 
for i=1:length(output) 
    if(test_targets(i)==1) 
        M=M+1;
    else 
        N=N+1;  
    end 
end 
sigma=M+N; %
for i=M+N:-1:1 
    if(test_targets(I(i))==1) 
        sigma=sigma+i; 
    end 
end
result=(sigma-(M-1)*M/2)/(M*N); 
end

