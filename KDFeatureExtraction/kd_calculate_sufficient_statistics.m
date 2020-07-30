function [N , F , cov_num , cov_denum ] = kd_calculate_sufficient_statistics( H , spect , cov_num , cov_denum, w_mean, num_of_basis, dim_of_feature )
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

%now updating the cov matrix

for c=1:num_of_basis
        cov_num(:,c)= cov_num(:,c)+sum(((spect-repmat(w_mean((c-1)*dim_of_feature+1:c*dim_of_feature,:),[1,size(spect,2)])).^2)* diag(posteriors(c,:)),2);
        cov_denum(:,c)=cov_denum(:,c)+sum(posteriors(c,:));
end

end
