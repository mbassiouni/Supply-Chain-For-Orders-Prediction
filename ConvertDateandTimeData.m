function [featureadded] = ConvertDateandTimeData()
load 'TrainData.mat';
for i = 1 : 700
   mycelldate(i) = ExcelSmallData(i,52);
   stringdate = mycelldate{i};
   valdate = stringdate;
   formatIn = 'dd/mmm/yyyy HH MM';
   finaldatavalue(i) = datenum(valdate,formatIn);
end
featureadded = finaldatavalue;
end