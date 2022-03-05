function [featureadded] = ConvertDateandTimeDataTest()
load 'TestData.mat';
for i = 1 : 300
   mycelldate(i) = ExcelSmallDataTest(i,52);
   stringdate = mycelldate{i};
   valdate = stringdate;
   formatIn = 'dd/mmm/yyyy HH MM';
   finaldatavalue(i) = datenum(valdate,formatIn);
end
featureadded = finaldatavalue;
end