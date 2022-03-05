function [featuresTrain,featuresTest] = ObtainFeaturesFromLSTM (XTrainSC,XTestSC,net)
idxLayer = 4;
for i = 1 : 700
   Xtrain = XTrainSC(i);  
   featuresTrain(i,:) = myActivations(net,Xtrain(:,1),idxLayer);
end
for i = 1 : 300
   Xtest = XTestSC(i);  
   %featuresTest(i,:) = activations(net,Xtrain(:,1),6);
   featuresTest(i,:) = myActivations(net,Xtest(:,1),idxLayer);
end

end