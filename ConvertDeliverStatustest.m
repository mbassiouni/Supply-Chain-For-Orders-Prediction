function [featureadded] = ConvertDeliverStatustest ()
load 'TestData.mat';
for i = 1 : 300
    featurecell(i) = ExcelSmallDataTest(i,6);
    feature(i) = featurecell{i};
    if strcmp(feature(i),"Late delivery")
        val(i) = 100;
    elseif strcmp(feature(i),"Advance shipping")
        val (i) = 101;
    elseif strcmp(feature(i),"Shipping canceled")
        val(i) = 102;
    elseif strcmp(feature(i),"Shipping on time")
        val(i) = 103;
    end
end
featureadded = val;
end