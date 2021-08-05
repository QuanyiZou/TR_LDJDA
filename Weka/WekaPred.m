function [r,t] = WekaPred(src,tar,type)
    WekaData2Arff('src.arff',src)
    WekaData2Arff('tar.arff',tar)
    Source = WekaInstances('src.arff');
    Target = WekaInstances('tar.arff');

    numAttr = Target.numAttributes;
    numInst = Target.numInstances;
    
    Measured = zeros(numInst,numAttr);
    for i=1:numAttr
        Measured(:,i) = Target.attributeToDoubleArray(i-1);
    end
    Measured = Measured(:,end);
    
    tic
    classifier = WekaClassifiers(type);
    Predicted = zeros(Target.numInstances,1); 
    classifier.buildClassifier(Source);
    t.train = toc;
    
    tic
    for i = 1:Target.numInstances
        Predicted(i) = classifier.classifyInstance(Target.instance(i-1));
    end
    t.test = toc;
    r.obs = Measured;
    r.pre = Predicted;
end

function D = WekaInstances(ArffFileAddress)
    loader = weka.core.converters.ArffLoader();
    loader.setFile(java.io.File(ArffFileAddress));
    D = loader.getDataSet();
    D.setClassIndex(D.numAttributes()-1);
end

function classifier = WekaClassifiers(Type)
    switch Type 
        case 'SMOreg'
            classifier = weka.classifiers.functions.SMOreg();
        case 'AdditiveRegression'
            classifier = weka.classifiers.meta.AdditiveRegression();
        case 'DecisionTable'
            classifier = weka.classifiers.rules.DecisionTable();
        case 'LeastMedSq'
            classifier = weka.classifiers.functions.LeastMedSq();
        case 'LinearRegression'
            classifier = weka.classifiers.functions.LinearRegression();
        case 'RBFNetwork'
            classifier = weka.classifiers.functions.RBFNetwork();
        case 'SimpleLinearRegression'
            classifier = weka.classifiers.functions.SimpleLinearRegression();
        case 'ZeroR'
            classifier = weka.classifiers.rules.ZeroR(); 
    end
end
