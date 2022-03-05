function [featureadded] = ConvertDeliverStatus ()
load 'TrainData.mat';
for i = 1 : 700
    featurecell(i) = ExcelSmallData(i,6);
    feature(i) = featurecell{i};
    if strcmp(feature(i),"Late delivery")
        val(i) = 1;
    elseif strcmp(feature(i),"Advance shipping")
        val (i) = 2;
    elseif strcmp(feature(i),"Shipping canceled")
        val(i) = 3;
    elseif strcmp(feature(i),"Shipping on time")
        val(i) = 4;
    end
end
featureadded = val;
end