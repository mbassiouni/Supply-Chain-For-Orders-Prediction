load 'DataNewAgain.mat';
%load 'NewNewFileData.mat';
[r,c] = size(ExcelSmallDataSmaller);
for i = 1 : r
   XTrainSC{i} = ExcelSmallDataSmaller(i,:)'; 
end
XTrainSC = XTrainSC';
[YTrainSC] = ConvertLabelsNumber_To_Categorial (TrainClasses);
YTrainSC = YTrainSC';
[r,c] = size(ExcelSmallDataSmallerTest);
for i = 1 : r
   XTestSC{i} = ExcelSmallDataSmallerTest(i,:)'; 
end
XTestSC = XTestSC';
[YTestSC] = ConvertLabelsNumber_To_Categorial (TestClasses);
YTestSC = YTestSC';
XX1 = categories(YTrainSC);
XX2 = categories(YTestSC);
inputSize = 13;
numHiddenUnits = 100; % best 100 with 9 features until now
numClasses = 2;
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(0.2)
    bilstmLayer(numHiddenUnits,'OutputMode','last') 
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]
%'LearnRateSchedule','piecewise',...
%'LearnRateDropFactor',0.2,...
%'LearnRateDropPeriod',5,...
maxEpochs = 20;
miniBatchSize = 27;
XTestSC = XTestSC';
ValidSC = XTestSC(1:36101);
ValidSClass = YTestSC(1:36101,1);
FinalTest = XTestSC(36102:end);
FinalTestclass = YTestSC(36102:end,1);
options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationData',{ValidSC,ValidSClass},...
    'ValidationFrequency',500,...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',1, ...
    'Plots','training-progress');
%YTrainSC = YTrainSC';
net = trainNetwork(XTrainSC,YTrainSC,layers,options);
%load 'netFinal.mat';
%load 'SavedModelNew.mat';
 %load 'OneConvolutionalLayer.mat';
% load 'TestClasses.mat';

 trainingFeatures = activations(convnet, Train_data, 6);
 testFeatures = activations(convnet, Test_data, 6);
 TestFinalFeatures = squeeze(testFeatures);
 TrainFinalFeatures = squeeze(trainingFeatures);
 TestFinalFeatures = TestFinalFeatures';
 TrainFinalFeatures = TrainFinalFeatures';
[AccKNN,GroupKNN] = KNNClassification(TrainFinalFeatures,TestFinalFeatures,TrainClasses,TestClasses);
trainingFeatures = activations(convnet, Train_data, 4);
testFeatures = activations(convnet, Test_data, 4);
TestFinalFeatures = squeeze(testFeatures);
TrainFinalFeatures = squeeze(trainingFeatures);
TestFinalFeatures = TestFinalFeatures';
TrainFinalFeatures = TrainFinalFeatures';
[AccSVM,GroupSVM] = SVMClassification(TrainFinalFeatures,TrainClasses,TestFinalFeatures,TestClasses,3);
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
