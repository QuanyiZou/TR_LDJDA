function [] = WekaData2Arff(fileName,data)
    type_run = 'reg';
    fid = fopen(fileName,'w');
    fprintf(fid,'@relation %s\n\n',fileName);
    [~,n] = size(data);
    for i=1:n-1
        fprintf(fid,'@attribute a%d numeric\n',i);
    end
    switch type_run
        case 'reg'
            fprintf(fid,'@attribute dT numeric\n');
        case 'cls'
            fprintf(fid,'@attribute class {0,1}\n');
    end
    fprintf(fid,'\n@data\n');
    str = mat2str(data);
    str = str(2:end-1);
    str = strrep(str,' ',',');
    str = strrep(str,';','\n');
    fprintf(fid,str);
    fclose(fid);
end