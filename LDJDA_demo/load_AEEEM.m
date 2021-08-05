clear all
addpath('../Weka');

addr = '../AEEEM/';
files = dir(addr);
Projects = cell(length(files)-2,1);
for i=3:length(files)
    name = [addr,files(i).name];
     Projects{i-2,1} = files(i).name;
    Projects{i-2,2} = WekaArff2Data(name);
end
