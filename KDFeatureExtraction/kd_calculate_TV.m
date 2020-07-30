function [T] = kd_calculate_TV(W_UBM_unfolded, W_UBM_varience_supervector, num_of_utterances, F_total, N_total)

disp(['Initializing T matrix (randomly)'])
dim_of_W_unfolded=length(W_UBM_unfolded);
%Choosing dimension of the ivector as 400
%T=randn(dim_of_W_unfolded,400);
T=randn(400,dim_of_W_unfolded); %KD change 19 june
S=[];  %not using second order stats, thus setting them to be empty 
n_sessions=num_of_utterances;
spk_id=[1:1:num_of_utterances];

%iteratively train T
for ii=1:10
  disp(' ')
  disp(['Starting iteration: ' num2str(ii)])
  
  ct=cputime;
  et=clock;
  
  [temp,T]=estimate_y_and_v(F_total, N_total, S, W_UBM_unfolded, W_UBM_varience_supervector, 0, T, 0, zeros(num_of_utterances,1), 0, zeros(n_sessions,1), spk_id);

  disp(['Iteration: ' num2str(ii) ' CPU time:' num2str(cputime-ct) ' Elapsed time: ' num2str(etime(clock,et))])
end

end 
  
