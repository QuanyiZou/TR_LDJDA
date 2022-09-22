# TR_LDJDA
# LDJDA-master
Source code and all the datasets in JDJDA.

## There are the following  main subfolders in -master
-- AEEEM, NASA, Promise, and Relink--
                   These folders include the corresponding datasets used in the paper. Each dataste is a ARFF file.
                   
-- DLJDA_method --  This folder includes the source code of DLJDA.

-- DLJDA_demo  ---  This folder includes four  demonstrations(eg.demo_AEEEM.m).

-- Full Reslut--    This document is the results of 511 cross-project defect prediction experiments.

-- R+Rank--         This folder includes the code of Scoott-Kntt ESD test with R.    

##  weka environment configuration 
-Step 1: Download and install weka;

-step 2: To check whether java is installed in the computer;

-Step 3：Copy the weka.jar package to the toolbox of matlab;

-Step 4: Enter "which classpath.txt" in the command line window of Matlab to open the "classpath.txt" file;

-Step 5: Add "$matlabroot/java/jar/toolbox/weka.jar" into classpath.txt

## Usage 
- Step1: Download TR_LDJDA-master;
- Step2: Open MATLAB, add the path.
- Step3: Add the absolute path of weka.jar into the classpath.txt of MATLAB.
- Step4: Enter folder  DLJDA_demo and open the script demo_AEEEM.m in MATLAB;
- Step5: Click the run button in the manu of MATLAB Editor to run the demo; 

## Contact
If you have any problem about our code, feel free to contact --zouquanyi2010@163.com or describe your problem in Issues.
