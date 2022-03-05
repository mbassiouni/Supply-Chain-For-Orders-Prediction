filename = "weatherReports.csv";
data = readtable(filename,'TextType','string');
data.event_narrative(1:700) = ExcelSmallData';
data.event_narrative(701:1000) = ExcelSmallDataTest' ;
data.event_narrative(1001:1700) = ExcelSmallData';
data.event_narrative(1701:2000) = ExcelSmallDataTest' ;
data.event_narrative(2001:2700) = ExcelSmallData';
data.event_narrative(2701:3000) = ExcelSmallDataTest' ;
head(data);
idxEmpty = strlength(data.event_narrative) == 0;
data(idxEmpty,:) = [];
data.event_type = categorical(data.event_type);
f = figure;
f.Position(3) = 1.5*f.Position(3);
h = histogram(data.event_type);
xlabel("Class")
ylabel("Frequency")
title("Class Distribution")
classCounts = h.BinCounts;
classNames = h.Categories;
idxLowCounts = classCounts < 10;
infrequentClasses = classNames(idxLowCounts);
idxInfrequent = ismember(data.event_type,infrequentClasses);
data(idxInfrequent,:) = [];
data.event_type = removecats(data.event_type);
cvp = cvpartition(data.event_type,'Holdout',0.1);
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);
textDataTrain = dataTrain.event_narrative;
textDataTest = dataTest.event_narrative;
YTrain = dataTrain.event_type;
YTest = dataTest.event_type;
textDataTrain = erasePunctuation(textDataTrain);
textDataTrain = lower(textDataTrain);
documentsTrain = tokenizedDocument(textDataTrain);
embeddingDimension = 1;
embeddingEpochs = 50;
emb = trainWordEmbedding(documentsTrain, ...
    'Dimension',embeddingDimension, ...
    'NumEpochs',embeddingEpochs, ...
    'Verbose',0);
xx = 10;
documentLengths = doclength(documentsTrain);
sequenceLength = 1;
documentsTruncatedTrain = docfun(@(words) words(1:min(sequenceLength,end)),documentsTrain);
XTrain = doc2sequence(emb,documentsTruncatedTrain);