function [Acc,GroupSVM] = SVMClassification(Train,Trainlabel,Test,Testlabel,val)
Mdl2 = fitcsvm(Train,Trainlabel);
[GroupSVM,scores] = predict(Mdl2,Test);
count = 0;
for i = 1 : length(GroupSVM)
   if GroupSVM(i) == Testlabel(i)
       count = count + 1;
   end
end
Acc = (count/ length(GroupSVM)) * 100;
end