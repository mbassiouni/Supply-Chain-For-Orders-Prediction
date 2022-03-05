s = load("HumanActivityTrain.mat");
XTrain = s.XTrain; % Train Features
TTrain = s.YTrain; % Train Labels
numObservations = numel(XTrain);
classes = categories(TTrain{1});
numClasses = numel(classes);
numFeatures = size(s.XTrain{1},1);
figure
for i = 1:3
    X = s.XTrain{1}(i,:);

    subplot(4,1,i)
    plot(X)
    ylabel("Feature " + i + newline + "Acceleration")
end

subplot(4,1,4)

hold on
plot(s.YTrain{1})
hold off

xlabel("Time Step")
ylabel("Activity")

subplot(4,1,1)
title("Training Sequence 1")
numFilters = 64;
filterSize = 3;
dropoutFactor = 0.005;
numBlocks = 4;

layer = sequenceInputLayer(numFeatures,'Normalization','rescale-symmetric','Name','input');
lgraph = layerGraph(layer);

outputName = layer.Name;
    count = 1; x = 1;
for i = 1:numBlocks
    dilationFactor = 2^(i-1);

    layers = [
        convolution2dLayer([1 filterSize],numFilters,'DilationFactor',dilationFactor,'Padding','same','Name',strcat('conv1_',num2str(count)))
        layerNormalizationLayer('Name',strcat('layernorm_',num2str(x)))
        dropoutLayer(dropoutFactor,'Name',strcat('layer_',num2str(x)))
        convolution2dLayer([1 filterSize],numFilters,'DilationFactor',dilationFactor,'Padding','same','Name',strcat('conv1d_',num2str(count)))
        layerNormalizationLayer('Name',strcat('layernorm_',num2str(x+1)))
        reluLayer('Name',strcat('relu_',num2str(count)))
        dropoutLayer(dropoutFactor,'Name',strcat('layer',num2str(x+1)))
        additionLayer(2,'Name',strcat('add_',num2str(count)))];

    % Add and connect layers.
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,strcat('conv1_',num2str(count)));

    % Skip connection.
    if i == 1
        % Include convolution in first skip connection.
        layer = convolution2dLayer([1 1],numFilters,'Name','convSkip');

        lgraph = addLayers(lgraph,layer);
        lgraph = connectLayers(lgraph,outputName,'convSkip');
        lgraph = connectLayers(lgraph,'convSkip',strcat('add_',num2str(i),'/in2'));
    else
        lgraph = connectLayers(lgraph,outputName,strcat('add_',num2str(count),'/in2'));
    end
    
    % Update layer output name.
    outputName = "add_" + count;
    
    count = count + 1;
    x = x + 2;
end

layers = [
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','soft')
    classificationLayer('Name','class')];
lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,outputName,'fc');
figure
plot(lgraph)
title("Temporal Convolutional Network");
options = trainingOptions("adam", ...
    'MaxEpochs',60, ...
    'MiniBatchSize',1, ...
    'Plots','training-progress',...
    'Verbose',1);
net = trainNetwork(XTrain,TTrain,lgraph,options);