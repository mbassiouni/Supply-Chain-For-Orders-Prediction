function Read_LearnLSTM_Classify_SOFTMax()
load 'TrainSmallerData.mat';
[r,c] = size(ExcelSmallDataSmaller);
for i = 1 : r
   XTrainSC{i} = ExcelSmallDataSmaller(i,:)'; 
end
load 'TrainClasses.mat'
%valueset = {'Latedelivery','Advanceshipping','Shippingontime','Shippingcanceled'};
% for i = 1 : 700
% Newclass{i} = cellstr(MyClasses(i));    
% end
%YTrainSC = Newclass';
%YTrainSC = categorical(MyClasses);
%YTrainSC = DealClasses();
[YTrainSC] = ConvertLabelsNumber_To_Categorial (TrainClasses);
YTrainSC = YTrainSC';
%YTrainSC = TrainClasses;
load 'TestClasses.mat';
[YTestSC] = ConvertLabelsNumber_To_Categorial (TestClasses);
YTestSC = YTestSC';
XTrainSC = XTrainSC';
numObservations = numel(XTrainSC);
for i=1:numObservations
    sequence = XTrainSC{i}';
    sequenceLengths(i) = size(sequence,2);
end
[sequenceLengths,idx] = sort(sequenceLengths);
XTrainSC = XTrainSC(idx);
YTrainSC = YTrainSC(idx);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Training%%%%%%%%%%%%%%%%%%%%
inputSize = 9;
numHiddenUnits = 150; % best 100 with 9 features until now
numClasses = 2;
%c2 = convolution2dlayer([1 30],400);
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','sequence')
   % dropoutLayer(0.2)
   % bilstmLayer(numHiddenUnits,'OutputMode','last')
   % dropoutLayer(0.2)
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
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');
%[XTestSC, YTestSC] = testLSTM ();
load 'TestSmallerData.mat';
[r,c] = size(ExcelSmallDataSmallerTest);
for i = 1 : r
   XTestSC{i} = ExcelSmallDataSmallerTest(i,:)'; 
end
XTestSC = XTestSC';
net = trainNetwork(XTrainSC',YTrainSC',layers,options);
%load 'netFinal.mat';
load 'netFinal94.mat';
[featuresTrain,featurestest] = ObtainFeaturesFromLSTM(XTrainSC,XTestSC,net);
[AccSVM,GroupSVM] = SVMClassification(featuresTrain,TrainClasses,featurestest,TestClasses,3);
[AccKNN,GroupKNN] = KNNClassification(featuresTrain,featurestest,TrainClasses,TestClasses);
miniBatchSize = 27;
YPred = classify(net,XTestSC, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
AccSoftmax = (sum(YPred == YTestSC)./numel(YTestSC)) * 100.0;
fprintf('The accuracy based on KNN : %.2f%c\n',AccKNN,'%');
fprintf('The accuracy based on Softmax : %.2f%c\n',AccSoftmax,'%');
fprintf('The accuracy based on SVM : %.2f%c\n',AccSVM,'%');
figure;
plotconfusion(YTestSC,YPred);
title('Confusion Matrix of Softmax','FontSize',20);
figure;
GroupSVM = categorical(GroupSVM);
plotconfusion(YTestSC,GroupSVM);
title('Confusion Matrix of SVM','FontSize',20);
figure;
GroupKNN = categorical(GroupKNN);
plotconfusion(YTestSC,GroupKNN);
title('Confusion Matrix of KNN','FontSize',20);
figure;
load 'RandomData.mat';
GroupRandomForest = categorical(RandomData);
plotconfusion(YTestSC,GroupRandomForest);
AccRandomForest = (sum(GroupRandomForest == YTestSC)./numel(YTestSC)) * 100.0;
title('Confusion Matrix of Random Forest','FontSize',20);
figure;
load 'RandomTree.mat';
GroupRandomTree = categorical(RandomTree);
plotconfusion(YTestSC,GroupRandomTree);
AccRandomTree = (sum(GroupRandomTree == YTestSC)./numel(YTestSC)) * 100.0;
title('Confusion Matrix of Random Tree','FontSize',20);
end