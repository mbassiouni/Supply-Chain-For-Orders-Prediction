function [featureadded] = ConvertDeliverStatus ()
load 'Data1.mat';
for i = 1 : 180519
   % featurecell(i) = ExcelSmallData(i,6);
    feature(i) = DataCoSupplyChainDataset(i);
    if strcmp(feature(i),"Late delivery")
        val(i) = 1;
    elseif strcmp(feature(i),"Advance shipping")
        val (i) = 0;
    elseif strcmp(feature(i),"Shipping canceled")
        val(i) = 0;
    elseif strcmp(feature(i),"Shipping on time")
        val(i) = 0;
    end
end
featureadded = val;
end