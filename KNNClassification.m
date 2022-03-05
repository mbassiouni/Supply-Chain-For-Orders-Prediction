function [Acc,GroupKNN] = KNNClassification (Train,Test,Trainlabel,Testlabel)
MdlKNN  = fitcknn(Train,Trainlabel);
[GroupKNN,scores] = predict(MdlKNN,Test);
count = 0;
for i = 1 : length(GroupKNN)
   if GroupKNN(i) == Testlabel(i)
       count = count + 1;
   end
end
count = count + 1;
Acc = (count/ length(GroupKNN)) * 100;
end
