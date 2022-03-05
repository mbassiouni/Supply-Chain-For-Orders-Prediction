%load 'TrainSmallerData.mat';
%XTrainSC = ExcelSmallDataSmaller'; 
load 'MyDDD.mat';
%load 'XTrainSC.mat';
XTrainSC = myTri;
XTestSC = myTes;
Train_data = reshape(XTrainSC, [1 14 1 6000]);
%load 'TrainClasses.mat';
[TrainC] = ConvertLabelsNumber_To_Categorial (TrainClasses);
inputLayer=imageInputLayer([1 14]);
c1=convolution2dLayer([1 12],16,'stride',2);
r1 = reluLayer;
b1 = batchNormalizationLayer;
p1=maxPooling2dLayer([1 2],'stride',1);
c2=convolution2dLayer([1 1],16,'stride',2);
r2 = reluLayer;
b2 = batchNormalizationLayer;
p2=maxPooling2dLayer([1 1],'stride',1);
c3=convolution2dLayer([1 1],16,'stride',2);
r3 = reluLayer;
b3 = batchNormalizationLayer;
f1=fullyConnectedLayer(2);
s1=softmaxLayer;
outputLayer=classificationLayer;
convnet=[inputLayer; c1; r1; b1; p1; c2; r2; b2; p2;c3; r3; b3; f1;s1;outputLayer]
analyzeNetwork(convnet);
analyzeNetwork(c1);
ilr = 0.001;
mxEpochs = 20;
mbsize = 10;
%'LearnRateSchedule','piecewise',...
%'LearnRateDropFactor',0.2,...
%'LearnRateDropPeriod',5,...
options = trainingOptions('sgdm',...
    'InitialLearnRate',ilr,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.2,... // 0.5 --->  94.4  ----->  97.22
    'LearnRateDropPeriod',5,...
    'MaxEpochs',mxEpochs,...
    'MiniBatchSize',mbsize,...
    'Plots','training-progress');
convnet = trainNetwork(Train_data,TrainC,convnet,options);
% load 'OneConvolutionalLayer.mat';
% load 'TestClasses.mat';
[YTestSC] = ConvertLabelsNumber_To_Categorial (TestClasses);
%load 'BestConvnet.mat';
% load 'TestSmallerData.mat';
% [r,c] = size(ExcelSmallDataSmallerTest);
% XTestSC = ExcelSmallDataSmallerTest'; 
Test_data = reshape(XTestSC, [1 14 1 4000]);
 trainingFeatures = activations(convnet, Train_data, 10);
 testFeatures = activations(convnet, Test_data, 10);
 TestFinalFeatures = squeeze(testFeatures);
 TrainFinalFeatures = squeeze(trainingFeatures);
 TestFinalFeatures = TestFinalFeatures';
 TrainFinalFeatures = TrainFinalFeatures';
% [AccKNN,GroupKNN] = KNNClassification(TrainFinalFeatures,TestFinalFeatures,TrainClasses,TestClasses);
% trainingFeatures = activations(convnet, Train_data, 4);
% testFeatures = activations(convnet, Test_data, 4);
% TestFinalFeatures = squeeze(testFeatures);
% TrainFinalFeatures = squeeze(trainingFeatures);
% TestFinalFeatures = TestFinalFeatures';
% TrainFinalFeatures = TrainFinalFeatures';
% [AccSVM,GroupSVM] = SVMClassification(TrainFinalFeatures,TrainClasses,TestFinalFeatures,TestClasses,3);
[predicatedlabels,scores] = classify(convnet,Test_data);
YTestSC = YTestSC';
AccSoftmax = mean(predicatedlabels == YTestSC) * 100;
fprintf('The accuracy based on KNN : %.2f%c\n',AccKNN,'%');
fprintf('The accuracy based on Softmax : %.2f%c\n',AccSoftmax,'%');
fprintf('The accuracy based on SVM : %.2f%c\n',AccSVM,'%');
%defining my colors
f1=[0 0 139]/255;
f4=[50 205 50]/255;
f9=[236 0 0]/255;
f14=[85 26 139]/255;
%example
a=[1 1 0 1 0 0 1 0 1 0 1 1 0 0 1 0 0 0 1 0];
b=[1 0 1 1 0 1 1 0 1 0 1 0 0 0 1 1 0 0 1 0];
figure;
plotconfusion(YTestSC,predicatedlabels);
title('Confusion Matrix of Softmax','FontSize',20);
xlabel('Target Class','FontSize',20,'FontWeight','Bold');
ylabel('Output Class','FontSize',20,'FontWeight','Bold');
%fontsize
set(findobj(gca,'type','text'),'fontsize',16) 
set(gca,'FontSize',20)
%colors          
set(findobj(gca,'color',[0,102,0]./255),'color',f4)
set(findobj(gca,'color',[102,0,0]./255),'color',f9)
set(findobj(gcf,'facecolor',[120,230,180]./255),'facecolor',f4)
set(findobj(gcf,'facecolor',[230,140,140]./255),'facecolor',f9)
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',f1)
set(findobj(gcf,'facecolor',[120,150,230]./255),'facecolor',f14)
figure;
plotconfusion(categorical(TestClasses),categorical(GroupSVM));
title('Confusion Matrix of SVM','FontSize',20);
xlabel('Target Class','FontSize',20,'FontWeight','Bold');
ylabel('Output Class','FontSize',20,'FontWeight','Bold');
%fontsize
set(findobj(gca,'type','text'),'fontsize',16) 
set(gca,'FontSize',20)
%colors          
set(findobj(gca,'color',[0,102,0]./255),'color',f4)
set(findobj(gca,'color',[102,0,0]./255),'color',f9)
set(findobj(gcf,'facecolor',[120,230,180]./255),'facecolor',f4)
set(findobj(gcf,'facecolor',[230,140,140]./255),'facecolor',f9)
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',f1)
set(findobj(gcf,'facecolor',[120,150,230]./255),'facecolor',f14)
figure;
plotconfusion(categorical(TestClasses),categorical(GroupKNN));
title('Confusion Matrix of KNN','FontSize',20);
xlabel('Target Class','FontSize',20,'FontWeight','Bold');
ylabel('Output Class','FontSize',20,'FontWeight','Bold');
%fontsize
set(findobj(gca,'type','text'),'fontsize',16) 
set(gca,'FontSize',20)
%colors          
set(findobj(gca,'color',[0,102,0]./255),'color',f4)
set(findobj(gca,'color',[102,0,0]./255),'color',f9)
set(findobj(gcf,'facecolor',[120,230,180]./255),'facecolor',f4)
set(findobj(gcf,'facecolor',[230,140,140]./255),'facecolor',f9)
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',f1)
set(findobj(gcf,'facecolor',[120,150,230]./255),'facecolor',f14)
load 'ANNConv.mat';
figure;
plotconfusion(categorical(TestClasses),categorical(ANN));
title('Confusion Matrix of ANN','FontSize',20);
xlabel('Target Class','FontSize',20,'FontWeight','Bold');
ylabel('Output Class','FontSize',20,'FontWeight','Bold');
%fontsize
set(findobj(gca,'type','text'),'fontsize',16) 
set(gca,'FontSize',20)
%colors          
set(findobj(gca,'color',[0,102,0]./255),'color',f4)
set(findobj(gca,'color',[102,0,0]./255),'color',f9)
set(findobj(gcf,'facecolor',[120,230,180]./255),'facecolor',f4)
set(findobj(gcf,'facecolor',[230,140,140]./255),'facecolor',f9)
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',f1)
set(findobj(gcf,'facecolor',[120,150,230]./255),'facecolor',f14)

load 'RandomForestConv.mat';
figure;
plotconfusion(categorical(TestClasses),categorical(RandomForest));
title('Confusion Matrix of Random Forest','FontSize',20);
xlabel('Target Class','FontSize',20,'FontWeight','Bold');
ylabel('Output Class','FontSize',20,'FontWeight','Bold');
%fontsize
set(findobj(gca,'type','text'),'fontsize',16) 
set(gca,'FontSize',20)
%colors          
set(findobj(gca,'color',[0,102,0]./255),'color',f4)
set(findobj(gca,'color',[102,0,0]./255),'color',f9)
set(findobj(gcf,'facecolor',[120,230,180]./255),'facecolor',f4)
set(findobj(gcf,'facecolor',[230,140,140]./255),'facecolor',f9)
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',f1)
set(findobj(gcf,'facecolor',[120,150,230]./255),'facecolor',f14)

load 'RandomTreeConv.mat';
figure;
plotconfusion(categorical(TestClasses),categorical(RandomTree));
title('Confusion Matrix of Random Tree','FontSize',20);
xlabel('Target Class','FontSize',20,'FontWeight','Bold');
ylabel('Output Class','FontSize',20,'FontWeight','Bold');
%fontsize
set(findobj(gca,'type','text'),'fontsize',16) 
set(gca,'FontSize',20)
%colors          
set(findobj(gca,'color',[0,102,0]./255),'color',f4)
set(findobj(gca,'color',[102,0,0]./255),'color',f9)
set(findobj(gcf,'facecolor',[120,230,180]./255),'facecolor',f4)
set(findobj(gcf,'facecolor',[230,140,140]./255),'facecolor',f9)
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',f1)
set(findobj(gcf,'facecolor',[120,150,230]./255),'facecolor',f14)