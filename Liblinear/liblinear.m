function [pre,dis] = liblinear(src,tar)
    model = train(src(:,end),sparse(src(:,1:end-1)),'-s 0 -B -1 -q');
    [pre,~,dis] = predict(tar(:,end),sparse(tar(:,1:end-1)),model,'-b -1 -q'); 
    dis = dis(:,1);
end