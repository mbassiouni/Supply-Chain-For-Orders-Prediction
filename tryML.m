load 'NewNewFileData.mat';
%[AccKNN,GroupKNN] = KNNClassification(ExcelSmallDataSmaller,ExcelSmallDataSmallerTest,TrainClasses,TestClasses);
FinalAllData = [ExcelSmallDataSmaller;ExcelSmallDataSmallerTest];
LabelsAll = [TrainClasses;TestClasses];
%load 'FinalFinalData.mat';
AllData = FinalAllData(1:180519,1:13); 
AllDataclasses = LabelsAll(1:180519);
%AllDataclasses = ConvertLabelsNumber_To_Categorial(AllDataclasses(1:1000));
num_folds = 5;
lengthoforders = length(AllDataclasses);
for fold_idx=1:num_folds
    test_idx=fold_idx:num_folds:lengthoforders;
    for i = 1 : length(test_idx)
         Foldtest(i,1:13) = AllData(test_idx(i),1:13); 
         Foldclassestest(i,1)= AllDataclasses(test_idx(i));
    end
    train_idx=setdiff(1:length(AllData),test_idx);
    for i = 1 : length(train_idx)
         Foldtrain(i,1:13) = AllData(train_idx(i),1:13); 
         Foldclassestrain(i,1)= AllDataclasses(train_idx(i));
    end
    %[Foldtrain,FoldValid] = splitEachLabel(Foldtrain,0.8);
    cv = cvpartition(size(Foldtrain,1),'HoldOut',0.25);
    idx = cv.test;
    FoldtrainFinal = Foldtrain(~idx,1:13);
    FoldtrainLabels = AllDataclasses(~idx);
    FoldValid  = Foldtrain(idx,1:13);
    FoldclassesValid = AllDataclasses(idx);
end
%[AccKNN,GroupKNN] = KNNClassification(ExcelSmallDataSmaller,ExcelSmallDataSmallerTest,TrainClasses,TestClasses);