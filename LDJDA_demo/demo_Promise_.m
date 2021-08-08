clear all
addpath('../DLJDA_method','../Weka','../Liblinear');
%load_AEEEM;  %% laod data;
load_Promise
%super-parameter
parameter.gamma=0.1;% 
parameter.lambda=0.1; % {10^3; 10^2; 10^1; 1; 10; 10^2; 10^3}.
parameter.layers=10; %{1,2,3,4,5,6,7,8,9,10}
parameter.noises=0.9; %{0.9}

dataset='Promise';  
modelName = 'LDJDA';
file_name=[modelName,'_',dataset,'_result.csv'];
file=fopen(file_name,'w'); 
headerStr ='model,task,source,target,pd,pf,f_measure,g_mesure,balance,MCC, AUC';
fprintf(file,['%s','\n'],headerStr);

file_nameav=[modelName,'_',dataset,'_average_result.csv']; %   
file_av=fopen(file_nameav,'w');
 av_headerStr ='model,project,av_pd,av_pf,av_f_measure,av_g_mesure,av_balance,av_MCC, av_AUC';
 fprintf(file_av,['%s','\n'],av_headerStr);
    
percent_tt = 0.9; % the percentage of training data in source data
runtimes=30;
num_pro=size(Projects,1);  % ÏîÄ¿µÄ¸öÊý

for i=1:num_pro
    tar_name=Projects{i,1}; % XX.arrf
    tar_name(end-4:end)=[]; % delete.arrf
    tar_data=CrossProjects{i,2}; 
    X_tar=tar_data(:,1:end-1);
    tar_num=size(tar_data,1);
    Y_tar=tar_data(:,end)';
    src=CrossProjects{i,1};
    src_Name=CrossProjects{i,3};
    Result_tar=[];  
    
    for j=1:size(src,1)
        src_name=src_Name{j};
        src_name(end-4:end)=[];
        task=strcat(src_name, '==>', tar_name) 
        src_data= src{j};
        Src_num=size(src_data,1);
        X_src=src_data(:,1:end-1);
        Y_src=src_data(:,end)';
        [X_src,X_tar] = normal(X_src,X_tar,'heuristic');       
        [X_src_new,X_tar_new,A]=mLDJDA(X_src,X_tar,Y_src,parameter);           
      
        Result=[];
        for m=1: runtimes
           idx=randperm(Src_num,round(percent_tt*Src_num));  % randomly select 90% the source projects as training data
           X_tran=X_src_new(:,idx);  % traning data X 
           Y_tran=Y_src(:,idx);   % traning data label           
            data_tran=[X_tran',Y_tran'];  % training data
           data_tar_new=[X_tar_new',Y_tar'];
           [pre,dis] = liblinear(data_tran,data_tar_new);  % prediction  target label 
           [pd1,pf1,f_measure1, g_mesure1,balance1,MCC1, AUC1] =  performance(pre', Y_tar,dis'); 
           result=[pd1,pf1,f_measure1, g_mesure1,balance1,MCC1, AUC1];
           Result=[Result;result]
        end
        reslut_med=median(Result);  %  Median of 30 runs
        pd=reslut_med(1);
        pf=reslut_med(2);
        f_measure= reslut_med(3); 
        g_mesure=reslut_med(4);
        balance=reslut_med(5);
        MCC=reslut_med(6);
        AUC=reslut_med(7);        
        resultStr =[modelName,',',task,',',src_name,',',tar_name,',',num2str(pd),',',num2str(pf),',',num2str(f_measure),',',num2str(g_mesure),',',num2str(balance),',',num2str(MCC),',',num2str(AUC),',';];
        fprintf(file,'%s\n',resultStr);  
        tar_result=[pd,pf,f_measure,g_mesure,balance,MCC, AUC]; %
        Result_tar=[Result_tar;tar_result];
    end 
    av_reslut=mean(Reslut);  %  The average value in a target project. 
    av_pd=av_reslut(1);
    av_pf=av_reslut(2);
    av_f_measure=av_reslut(3); 
    av_g_mesure=av_reslut(4);
    av_balance=av_reslut(5);
    av_MCC=av_reslut(6);
    av_AUC=av_reslut(7);      
    
    resultStr_av =[modelName,',',tar_name,',',num2str(av_pd),',',num2str(av_pf),',',num2str(av_f_measure),',',num2str(av_g_mesure),',',num2str(av_balance),',',num2str(av_MCC),',',num2str(av_AUC),',';];
    fprintf(file_av,'%s\n',resultStr_av);
end

