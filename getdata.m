function [MCC, kappa_overall] = getdata (TP,FP,FN,TN,accuracy)
%88,8,178,26
P=TP+FN;
N=FP+TN;
SEN = TP/(TP+FN);
fprintf('SEN = %f\n',SEN);
SPEC = TN/(TN+FP);
fprintf('SPEC = %f\n',SPEC);
Per = TP/(TP+FP);
fprintf('Per = %f\n',Per);
A = (TP+TN)/(TP+TN+FN+FP);
fprintf('Accuracy = %f\n',A);
FPR = FP/(FP+TN);
fprintf('FPR = %f\n',FPR);
FNR = FN/(FN+TP);
fprintf('FNR = %f\n',FNR);
Recall = SEN;
F1 = 2 * ((Per * Recall)/(Per + Recall));
fprintf('F1-measure = %f\n',F1);
Error = 1 - (A);
fprintf('Error = %f\n',Error);
MCC=[( TP.*TN - FP.*FN ) ./ ( ( (TP+FP).*P.*N.*(TN+FN) ).^(0.5) );...
                ( FP.*FN - TP.*TN ) ./ ( ( (TP+FP).*P.*N.*(TN+FN) ).^(0.5) )] ;
            MCC=max(MCC);
fprintf('MCC = %f\n',MCC);
            %Kappa Calculation BY 2x2 Matrix Shape
            pox=sum(accuracy);
            Px=sum(P);TPx=sum(TP);FPx=sum(FP);TNx=sum(TN);FNx=sum(FN);Nx=sum(N);
            pex=( (Px.*(TPx+FPx))+(Nx.*(FNx+TNx)) ) ./ ( (TPx+TNx+FPx+FNx).^2 );
            kappa_overall=([( pox-pex ) ./ ( 1-pex );( pex-pox ) ./ ( 1-pox )]);
            kappa_overall=max(kappa_overall);
fprintf('Kappa = %f\n',kappa_overall);
end