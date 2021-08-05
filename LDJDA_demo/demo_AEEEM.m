clear all
addpath('../DLJDA_method','../Weka','../Liblinear');
load_AEEEM;  %% laod data;

%super-parameter
parameter.gamma=0.1;% 
parameter.lambda=0.1; % {10^3; 10^2; 10^1; 1; 10; 10^2; 10^3}.
parameter.layers=10; %{1,2,3,4,5,6,7,8,9,10}
parameter.noises=0.9; 

dataset='AEEEM';  
modelName = 'LDJDA';  
file_name=[modelName,'_',dataset,'_result.csv'];
file=fopen(file_name,'w');
headerStr ='model,task,source,target,pd,pf,f_measure,g_mesure,balance,MCC, AUC';
fprintf(file,['%s','\n'],headerStr);

 file_nameav=[modelName,'_',dataset,'_average_result.csv'];  
 file_av=fopen(file_nameav,'w');
 av_headerStr ='model,project,av_pd,av_pf,av_f_measure,av_g_mesure,av_balance,av_MCC, av_AUC';
 fprintf(file_av,['%s','\n'],av_headerStr); 
    
percent_tt = 0.9; % the percentage of training data in source data
runtimes=20;    
num_pro=size(Projects,1);  % 项目的个数
index=[1:num_pro];

for i=1:num_pro
    tar_name=Projects{i,1}; % xxx.arrf
    tar_name(end-4:end)=[]; % delete.arrf
    tar_data=Projects{i,2};  
    X_tar=tar_data(:,1:end-1);
    tar_num=size(tar_data,1);
    Y_tar=tar_data(:,end)';
    index_1=find(index~=i);
    Result_tar=[];  
  
    for j=1:num_pro-1
        src_name=Projects{index_1(j),1};
        src_name(end-4:end)=[];
        task=strcat(src_name, '==>', tar_name)
        src_data=Projects{index_1(j),2};
        Src_num=size(src_data,1);
        X_src=src_data(:,1:end-1);
        Y_src=src_data(:,end)';
        [X_src,X_tar] = normal(X_src,X_tar,'heuristic');
        [X_src_new,X_tar_new,A]=mLDJDA(X_src,X_tar,Y_src,parameter);
        %f_measure=0; g_mesure=0;balance=0;MCC=0; AUC=0; pd=0;pf=0;
         Result=[];
        for m=1: runtimes   % run 30 times
            idx=randperm(Src_num,round(percent_tt*Src_num));  % randomly select 90% the source projects as training data
            X_tran=X_src_new(:,idx);  % traning data X 
            Y_tran=Y_src(:,idx);   % traning data label
            data_tran=[X_tran',Y_tran'];  % training data
            data_tar_new=[X_tar_new',Y_tar']; % test data
            [pre,dis] = liblinear(data_tran,data_tar_new);  % prediction  target label 
            [pd1,pf1,f_measure1, g_mesure1,balance1,MCC1, AUC1] =  performance(pre', Y_tar,dis');  % performance 
            result=[pd1,pf1,f_measure1, g_mesure1,balance1,MCC1, AUC1];
            Result=[Result;result];   
                 
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
    av_result=mean(Result_tar);  %  The average value in a target project. 
    av_pd=av_result(1);
    av_pf=av_result(2);
    av_f_measure=av_result(3); 
    av_g_mesure=av_result(4);
    av_balance=av_result(5);
    av_MCC=av_result(6);
    av_AUC=av_result(7);   
    resultStr_av =[modelName,',',tar_name,',',num2str(av_pd),',',num2str(av_pf),',',num2str(av_f_measure),',',num2str(av_g_mesure),',',num2str(av_balance),',',num2str(av_MCC),',',num2str(av_AUC),',';];
    fprintf(file_av,'%s\n',resultStr_av);
end
