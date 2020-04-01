%Title page

%Modeling of trees failure under windstorm in harvested forests using machine learning techniques

%Ali Jahani1*, Maryam Saffariha2

%* Corresponding author
%1*- Associate Prof., Faculty of Natural Environment and Biodiversity Department, College of Environment, Karaj, Iran.
%Email: Ajahani@ut.ac.ir

%2- Ph.D in Rangeland Management, College of Natural Resources, University of Tehran, Tehran, Iran
%Email: Saffariha@ut.ac.ir
%Email: Saffariha@ut.ac.ir

%%MLP MATLAB codes;
%Load, Divide Data
p=xlsread('inputs.xlsx');
t=xlsread('outputs.xlsx');
p = p';  %spectra 
t = t'; %Y 
[pn,pp1] = mapstd(p);
[R1,Q1] = size(pn)
[R,Q] = size(pn) 
iitr = [1:5:Q 3:5:Q 5:5:Q];
iitst = 2:5:Q;
iival = 4:5:Q;
vv.P = pn(:,iival); vv.T = t(:,iival);
vt.P = pn(:,iitst); vt.T = t(:,iitst);
ptr = pn(:,iitr); ttr = t(:,iitr);
inputs = ptr;
targets = ttr;
%Train Network
net1 = newpr(ptr,ttr,[5],{ 'purelin'});
[net1,tr] = train(net1,inputs,targets,[],[],vv,vt);
outputs = net1(inputs);
errors = gsubtract(targets,outputs);
performance1 = perform(net1,targets,outputs);

%%RBFNN MATLAB codes;
clc;
clear;
close all;
%Load Data 
x= xlsread('inputs.xlsx');

y= xlsread('outputs.xlsx');
 
inputs = x';
targets = y';
 
nData=size(inputs,2);
 
Perm=randperm(nData);
 
pTrainData=0.7;
nTrainData=round(pTrainData*nData);
trainInd=Perm(1:nTrainData);
Perm(1:nTrainData)=[];
trainInputs = inputs(:,trainInd);
trainTargets = targets(:,trainInd);
 
pTestData=1-pTrainData;
nTestData=nData-nTrainData;
testInd=Perm;
testInputs = inputs(:,testInd);
testTargets = targets(:,testInd);
 
 
% Create and Train RBF Network
Goal=0;
Spread=200;
MaxNeuron=30;
DisplayAt=1;
net = newrb(trainInputs,trainTargets,Goal,Spread,MaxNeuron,DisplayAt);
 
% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);
 
% Recalculate Training, Validation and Test Performance
trainOutputs = outputs(:,trainInd);
trainErrors = trainTargets-trainOutputs;
trainPerformance = perform(net,trainTargets,trainOutputs);
 
testOutputs = outputs(:,testInd);
testError = testTargets-testOutputs;
testPerformance = perform(net,testTargets,testOutputs);
 
PlotResults(targets,outputs,'All Data');
PlotResults(trainTargets,trainOutputs,'Train Data');
PlotResults(testTargets,testOutputs,'Test Data');
 
% View the Network
% view(net);


%%SVM MATLAB codes;

clc;
clear;
close all;
 
% Load Data
 
xall=xlsread('inputs.xlsx');
yall=xlsread('outputs.xlsx');
xtr=xlsread('inputstrain.xlsx');
ytr=xlsread('outputstrain.xlsx');
xtst=xlsread('inputstest.xlsx');
ytst=xlsread('outputstest.xlsx');
 
 TrainInputs=xtr;
TrainTargets=ytr;
 
n=numel(TrainTargets);
 
% Design SVM
 
C=20;
 
svmstruct=svmtrain(TrainInputs,TrainTargets,...
    'boxconstraint',C,...
    'kernel_function','rbf',...
    'rbf_sigma',10,...
    'polyorder',2,...
    'mlp_params',[1 -1],...
    'showplot',true);
 
nTrainData=480;
 
TrainOutputs=svmclassify(svmstruct,TrainInputs,'showplot',true);
plot (TrainOutputs,TrainTargets); 
 
mseTrain = mse(TrainOutputs,TrainTargets)
 
TrainOutputs=TrainOutputs';
TrainTargets=TrainTargets';
 
[cTr,cmTr,indTr,perTr] = confusion(TrainOutputs,TrainTargets);
 
 
% Test SVM
 
nTestData=120;
 
TestInputs=xtst;
 
TestTargets=ytst;
 
 
 
TestOutputs=svmclassify(svmstruct,TestInputs,'showplot',true);
plot (TestOutputs,TestTargets); 
 
mseTest = mse(TestOutputs,TestTargets)
 
TestOutputs=TestOutputs';
TestTargets=TestTargets';
 
[c,cm,ind,per] = confusion(TestOutputs,TestTargets);
 AllInputs=xall;
AllTargets=yall;
nAllData=600;
 
AllOutputs=svmclassify(svmstruct,AllInputs,'showplot',true);
plot (AllOutputs,AllTargets); 
 
mseAll = mse(AllOutputs,AllTargets)
 
AllOutputs=AllOutputs';
AllTargets=AllTargets';
 
[cAll,cmAll,indAll,perAll] = confusion(AllOutputs,AllTargets);




