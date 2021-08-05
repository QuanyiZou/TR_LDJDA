addpath('../Weka');

addr = '../Promise/';
files = dir(addr);
Projects = cell(length(files)-2,1);
for i=3:length(files)
    name = [addr,files(i).name];
    Projects{i-2,1} = files(i).name;
    Projects{i-2,2} = WekaArff2Data(name);
end

%% all combinations of cross-project predictions
load('Project_index.mat'); 
CrossProjects = cell(23,3);
for i=1:length(Projects)
    target_id = Project_index(i,1);
    target_project = Projects{i,2};
    source_id = find(Project_index(:,1)~=target_id(1));
    source_projects = Projects(Project_index(:,1)~=target_id(1),:);
    CrossProjects{i,1} = source_projects(:,2);    
    CrossProjects{i,2} = target_project;          
    CrossProjects{i,3} = source_projects(:,1);  
end