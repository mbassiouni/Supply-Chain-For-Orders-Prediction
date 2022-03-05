gpuDevice(1);
% load 'TrainSmallerData.mat';
% [r,c] = size(ExcelSmallDataSmaller); 
% XTrainSC{1} = ExcelSmallDataSmaller';
% load 'TestSmallerData.mat';
% [r,c] = size(ExcelSmallDataSmallerTest);
% XTestSC{1} = ExcelSmallDataSmallerTest'; 
% load 'TrainClasses.mat';
% [YTrainCC] = ConvertLabelsNumber_To_Categorial (TrainClasses);
% YTrainSC{1} = YTrainCC;
% load 'TestClasses.mat';
% [YTestCC] = ConvertLabelsNumber_To_Categorial (TestClasses);
% YTestSC{1} = YTestCC;
load 'TrainAll.mat';
load 'TestAll.mat'; %419430  %1887436
% [r,c] = size(ExcelSmallDataSmaller); 
XTrainSC{1} = TrainData(1:1048575,1:14)';
% load 'TestSmallerData.mat';
% [r,c] = size(ExcelSmallDataSmallerTest);
XTestSC{1} = TestData(1048576:1468006,1:14)'; 
% load 'TrainClasses.mat';
[YTrainCC] = ConvertLabelsNumber_To_Categorial (TrainClasses(1:1048575,1));
YTrainSC{1} = YTrainCC;
% load 'TestClasses.mat';
[YTestCC] = ConvertLabelsNumber_To_Categorial (TestClasses(1048576:1468006,1));
YTestSC{1} = YTestCC;
% YTestSC = YTestSC';
% XTrainSC = XTrainSC';
% XTestSC = XTestSC';
numObservations = numel(XTrainSC);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numBlocks = 1;
numFilters = 175;
filterSize = 3;
dropoutFactor = 0.05;
hyperparameters = struct;
hyperparameters.NumBlocks = numBlocks;
hyperparameters.DropoutFactor = dropoutFactor;
numInputChannels = 14;
parameters = struct;
numChannels = numInputChannels;
for k = 1:numBlocks
    parametersBlock = struct;
    blockName = "Block"+k;
    
    weights = initializeGaussian([filterSize, numChannels, numFilters]);
    bias = zeros(numFilters, 1, 'single');
    parametersBlock.Conv1.Weights = dlarray(weights);
    parametersBlock.Conv1.Bias = dlarray(bias);
    
    weights = initializeGaussian([filterSize, numFilters, numFilters]);
    bias = zeros(numFilters, 1, 'single');
    parametersBlock.Conv2.Weights = dlarray(weights);
    parametersBlock.Conv2.Bias = dlarray(bias);
    
    % If the input and output of the block have different numbers of
    % channels, then add a convolution with filter size 1.
    if numChannels ~= numFilters
        weights = initializeGaussian([1, numChannels, numFilters]);
        bias = zeros(numFilters, 1, 'single');
        parametersBlock.Conv3.Weights = dlarray(weights);
        parametersBlock.Conv3.Bias = dlarray(bias);
    end
    numChannels = numFilters;
    
    parameters.(blockName) = parametersBlock;
end
classes = categories(YTrainSC{1});
numClasses = numel(classes);
weights = initializeGaussian([numClasses,numChannels]);
bias = zeros(numClasses,1,'single');
parameters.FC.Weights = dlarray(weights);
parameters.FC.Bias = dlarray(bias);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maxEpochs = 400;
miniBatchSize = 1;
initialLearnRate = 0.1; % 0.9613
learnRateDropFactor = 0.9;
learnRateDropPeriod = 5;
gradientThreshold = 1;
iteration = 0;
executionEnvironment = "auto";
plots = "training-progress";
learnRate = initialLearnRate;
trailingAvg = [];
trailingAvgSq = [];
if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end
numObservations = numel(XTrainSC);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
start = tic;
% Loop over epochs.
for epoch = 1:maxEpochs
    
    % Shuffle the data.
    idx = randperm(numObservations);
    XTrain = XTrainSC(idx);
    YTrain = YTrainSC(idx);
    
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        % Read mini-batch of data and apply the transformSequences
        % preprocessing function.
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        [X,Y,numTimeSteps] = transformSequences(XTrain(idx),YTrain(idx));
        
        % Convert to dlarray.
        dlX = dlarray(X);
        
        % If training on a GPU, convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        
      
        % Evaluate the model gradients and loss using dlfeval.
        [gradients, loss] = dlfeval(@modelGradients,dlX,Y,parameters,hyperparameters,numTimeSteps);
        
        % Clip the gradients.
        gradients = dlupdate(@(g) thresholdL2Norm(g,gradientThreshold),gradients);
        
        % Update the network parameters using the Adam optimizer.
        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
            trailingAvg, trailingAvgSq, iteration, learnRate);
        
        if plots == "training-progress"
            % Plot training progress.
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            
            % Normalize the loss over the sequence lengths            
            loss = mean(loss ./ numTimeSteps);
            loss = double(gather(extractdata(loss)));
            loss = mean(loss);
            
            addpoints(lineLossTrain,iteration, mean(loss));

            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
    
    % Reduce the learning rate after learnRateDropPeriod epochs
    if mod(epoch,learnRateDropPeriod) == 0
        learnRate = learnRate*learnRateDropFactor;
    end
end
numObservationsTest = numel(XTestSC);
t = 10;
[X,Y] = transformSequences(XTestSC,YTestSC);
dlXTest = dlarray(X);
dlXrain = dlarray(dlX);
featurestrain = squeeze(dlXrain);
myFtrain = extractdata(featurestrain);
myFtrain = gather(myFtrain);
myFtrain = double(myFtrain);
featurestest = squeeze(dlXTest);
myFtest = extractdata(featurestest);
myFtest = double(myFtest);
[AccSVM,GroupSVM] = SVMClassification(myFtrain',TrainClasses(1:6000),myFtest',TestClasses(1:4000),3);
[AccKNN,GroupKNN] = KNNClassification(myFtrain',myFtest',TrainClasses(1:6000),TestClasses(1:4000));
doTraining = true;
dlYPred = model(dlXTest,parameters,hyperparameters,doTraining);
YPred = gather(extractdata(dlYPred));
labelsPred = categorical(zeros(numObservationsTest,size(dlYPred,3)));
accuracy = zeros(1,numObservationsTest);
for i = 1:numObservationsTest
    [~,idxPred] = max(YPred(:,i,:),[],1);
    [~,idxTest] = max(Y(:,i,:),[],1);
    labelsPred(i,:) = classes(idxPred)';
    accuracy(i) = mean(idxPred == idxTest);
end
mean(accuracy);
xxxx = 1000;