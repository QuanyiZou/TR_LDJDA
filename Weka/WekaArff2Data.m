function [data] = WekaArff2Data(fName)
    Target = WekaInstances(fName);

    numAttr = Target.numAttributes;
    numInst = Target.numInstances;
    
    data = zeros(numInst,numAttr);
    for i=1:numAttr
        column = Target.attributeToDoubleArray(i-1);
        for j=1:numInst
            data(j,i) = column(j);
        end
    end
    data(data(:,end)==0,end)=-1;
end

function D = WekaInstances(ArffFileAddress)
    loader = weka.core.converters.ArffLoader();
    loader.setFile(java.io.File(ArffFileAddress));
    D = loader.getDataSet();
    D.setClassIndex(D.numAttributes()-1);
end