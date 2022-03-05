load 'AllData.mat';
%load 'OldData.mat';
%load 'NewFileData.mat';
FinalAllData = [TrainData;TestData];
LabelsAll = [TrainClasses;TestClasses];
%load 'FinalFinalData.mat';
AllData = FinalAllData(1:4547660,1:14); 
AllDataclasses = ConvertLabelsNumber_To_Categorial(LabelsAll(1:4547660));
%AllDataclasses = ConvertLabelsNumber_To_Categorial(AllDataclasses(1:1000));
num_folds = 5;
lengthoforders = length(AllDataclasses);
for fold_idx=1:num_folds
    test_idx=fold_idx:num_folds:lengthoforders;
    for i = 1 : length(test_idx)
         Foldtest(i,1:14) = AllData(test_idx(i),1:14); 
         Foldclasses(i,1)= AllDataclasses(test_idx(i));
    end
    train_idx=setdiff(1:length(AllData),test_idx);
    for i = 1 : length(train_idx)
         Foldtrain(i,1:14) = AllData(train_idx(i),1:14); 
         Foldclassestrain(i,1)= AllDataclasses(train_idx(i));
    end
    %[Foldtrain,FoldValid] = splitEachLabel(Foldtrain,0.8);
    cv = cvpartition(size(Foldtrain,1),'HoldOut',0.25);
    idx = cv.test;
    FoldtrainFinal = Foldtrain(~idx,1:14);
    FoldtrainLabels = AllDataclasses(~idx);
    FoldValid  = Foldtrain(idx,1:14);
    FoldValidLabels = AllDataclasses(idx);
   % FoldtrainLabels = FoldtrainLabels';
    FoldtrainFinal = FoldtrainFinal';
    FoldValid = FoldValid';
    for i = 1 : length(FoldtrainLabels)
        FoldtrainFinal(:,i) = FoldtrainFinal(:,i)';
     XTrainSC{i} = FoldtrainFinal(:,i); 
    end
    for i = 1 : length(FoldValidLabels)
        FoldValid(:,i) = FoldValid(:,i)';
     XValidSC{i} = FoldValid(:,i); 
    end
   % Train_data = reshape(FoldtrainFinal, [1 9 1 60000]);
   % Valid_data = reshape(FoldValid, [1 9 1 20000]);
   % inputLayer=imageInputLayer([1 9]);
inputSize = 14;
numHiddenUnits = 150; % best 100 with 9 features until now
numClasses = 2;
%c2 = convolution2dlayer([1 30],400);
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(0.2)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]
%analyzeNetwork(layers);
% layers = [ ...
%     sequenceInputLayer(inputSize)
%     %sequenceFoldingLayer
%    % convolution2dLayer(filterSize,numFilters,'Name','conv')
%     %convolution2dLayer(9,7)
%     reluLayer
%     bilstmLayer(numHiddenUnits,'OutputMode','last')
%    % lstmLayer(numHiddenUnits,'OutputMode','last')
%     fullyConnectedLayer(numClasses)
%     dropoutLayer(0.2)
%     softmaxLayer
%     classificationLayer]
maxEpochs = 100;
miniBatchSize = 27;
options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationData',{XValidSC,FoldValidLabels'}, ...
    'Verbose',true, ...
    'ValidationFrequency',30,...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',1, ...
    'Plots','training-progress');
FoldtrainLabels = FoldtrainLabels';
net = trainNetwork(XTrainSC,FoldtrainLabels,layers,options);
%FoldtrainLabels = reshape(FoldtrainLabels, [1 1 1 6000]);
%convnet = trainNetwork(Train_data,FoldtrainLabels,convnet,options);
 end
%[AllDataclasses] = ConvertLabelsNumber_To_Categorial (TrainClasses);
%load 'MyDDD.mat';
load 'XTrainSC.mat';
%XTrainSC = myTri;
%XTestSC = myTes;
Train_data = reshape(XTrainSC, [1 9 1 700]);
load 'TrainClasses.mat';
[TrainC] = ConvertLabelsNumber_To_Categorial (TrainClasses);
inputLayer=imageInputLayer([1 9]);
c1=convolution2dLayer([1 8],100,'stride',1);
p1=maxPooling2dLayer([1 2],'stride',100);
c2=convolution2dLayer([1 1],100);
p2=maxPooling2dLayer([1 1],'stride',100);
f1=fullyConnectedLayer(2);
s1=softmaxLayer;
outputLayer=classificationLayer;
convnet=[inputLayer; c1; p1; c2; p2; f1;s1;outputLayer]
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
%convnet = trainNetwork(Train_data,TrainC,convnet,options);
 load 'OneConvolutionalLayer.mat';
 load 'TestClasses.mat';
[YTestSC] = ConvertLabelsNumber_To_Categorial (TestClasses);
%load 'BestConvnet.mat';
 load 'TestSmallerData.mat';
 [r,c] = size(ExcelSmallDataSmallerTest);
 XTestSC = ExcelSmallDataSmallerTest'; 
Test_data = reshape(XTestSC, [1 9 1 300]);
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
