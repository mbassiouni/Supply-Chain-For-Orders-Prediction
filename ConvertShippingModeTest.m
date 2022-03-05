function [featureadded] = ConvertShippingModeTest()
load 'TestData.mat';
for i = 1 : 300
    featurecell(i) = ExcelSmallDataTest(i,53);
    feature(i) = featurecell{i};
    if strcmp(feature(i),"First Class")
        val(i) = 20;
    elseif strcmp(feature(i),"Second Class")
        val (i) = 21;
    elseif strcmp(feature(i),"Standard Class")
        val(i) = 22;
    elseif strcmp(feature(i),"Same Day")
        val(i) = 23;
    end
end
featureadded = val;
end