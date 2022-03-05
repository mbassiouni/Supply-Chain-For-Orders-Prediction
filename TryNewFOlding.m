load 'NewNewFileData.mat';
Fold1 = ExcelSmallDataSmallerTest(1:36104,1:13);
labelFold1 = TestClasses(1:36104);
TrainFold1 = ExcelSmallDataSmaller;
labelTrainFold1 = TrainClasses;

Fold2 = [ExcelSmallDataSmallerTest(36105:end,1:13);ExcelSmallDataSmaller(1,1:13)];
labelFold2 = [TestClasses(36105:end,1);TrainClasses(1,1)];
TrainFold2 = [ExcelSmallDataSmaller(2:end,1:13);ExcelSmallDataSmallerTest(1,1:13)];
labelTrainFold2 = [TrainClasses(2:end,1);TestClasses(1,1)];

Fold3 = ExcelSmallDataSmaller(2:36105,1:13);
labelFold3 = TrainClasses(2:36105);
TrainFold3 = [ExcelSmallDataSmaller(36105:end,1:13);ExcelSmallDataSmallerTest(2:36105,1:13)];
labelTrainFold3 = [TrainClasses(36105:end,1);TestClasses(2:36105,1)];

Fold4 = ExcelSmallDataSmaller(36106:72209,1:13);
labelFold4 = TrainClasses(36106:72209);
TrainFold4 = [ExcelSmallDataSmaller(72208:end,1:13);ExcelSmallDataSmallerTest(1:end,1:13)];
labelTrainFold4 = [TrainClasses(72208:end,1);TestClasses(1:end,1)];

Fold5 = ExcelSmallDataSmaller(72210:end,1:13);
labelFold5 = TrainClasses(72210:end);
TrainFold5 = [ExcelSmallDataSmaller(1:72208,1:13);ExcelSmallDataSmallerTest(1:36104,1:13)];
labelTrainFold5 = [TrainClasses(1:72208,1);TestClasses(1:36104,1)];


% L = 36104+36103+36104;
% TestData = ExcelSmallDataSmaller(72207:L,1:13);
% Testlabel = TrainClasses(72207:L,1);
% L1 = ExcelSmallDataSmaller(1:36104:end,1:13);
% LL1 = TrainClasses(1:36104:end,1);
% TrainData = [L1 ; ExcelSmallDataSmaller(L+1:end,1:13); ExcelSmallDataSmallerTest(4:end,1:13)] ;
% Trainlabel = [LL1 ;TrainClasses(L+1:end,1);TestClasses(4:end,1)];
