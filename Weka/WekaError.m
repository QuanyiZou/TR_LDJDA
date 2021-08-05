function [f1,pofb20] = WekaError(obs,pre,dis,line)
    %% confusing matrix
    TP = sum(obs==1  & pre==1);
    FP = sum(obs==1  & pre==-1);
    FN = sum(obs==-1 & pre==1);

    %% f1-score
    precision = TP / (TP + FP);
    recall    = TP / (TP + FN);
    f1 = 2*precision*recall / (precision+recall);
    f1(isnan(f1))=0;
    
    %% pofb20
    obs(obs==-1) = 0;
    list = sortrows([dis,line,obs],-1);
    clist = cumsum(list(:,2))./sum(list(:,2));
    pofb20 = sum(list(1:find(clist>=0.2,1),3));
    pofb20 = pofb20/sum(obs);
end