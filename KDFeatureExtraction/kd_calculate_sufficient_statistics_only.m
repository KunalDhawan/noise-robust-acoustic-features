function [N , F ] = kd_calculate_sufficient_statistics_only( H , spect )
[rows_H, ~] = size(H);
sum_of_each_column=sum(H);
denom=repmat(sum_of_each_column,[rows_H,1]);
posteriors=H ./ denom;
posteriors(denom==0)=0;  % for the cases of division by 0

[rows_spec, ~] = size(spect);
N = sum(posteriors,2);     %0th order statistic
F = spect * posteriors' ;
% rows_h=number of basis elements && rows_spec= dimension of feature
F = reshape(F , rows_H * rows_spec, 1);

%following step is carried out to bring the stats to the format the code
%expects them:
N=N';
F=F';

%Final shape of N =( number of basis elements( equivalent to number of
%components in a GMM) X 1 )'
%final shape of F = (number of basis elements*dim of feature X 1)' 

end
