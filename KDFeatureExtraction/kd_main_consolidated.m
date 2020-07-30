%kd_main_consolidated

% %Initialising the important variables!
disp('Initializing the parameters !') 

%Dictionary params
num_of_basis_elements=60;
context_len=1;
Wsparsity=0;
Hsparsity=2;
batch_size=150;

%pre-emphasis variables
apre = [1 0.97];
bpre = [1];

%Filelist params
datadir = '/easyshare/kunal/aurora4';
filelist = '/easyshare/kunal/aurora4/lists/my_lists/training_multicondition_16k_sorted.list';
infile = fopen(filelist);
files = textscan(infile, '%s');
files = files{:};
fclose(infile);

% Spectrogram parameters
nfft = 1024;  % nfft-point DFT
win_len = 0.025;
win_shift = 0.01;

%Dimension params
dim_of_features=(nfft/2)+1;
num_of_utterances=length(files);

outdir = ['kd_features_',num2str(num_of_basis_elements),'_',num2str(context_len),'_',num2str(Wsparsity),'_',num2str(Hsparsity)];
outdir_log = ['kd_features_',num2str(num_of_basis_elements),'_',num2str(context_len),'_',num2str(Wsparsity),'_',num2str(Hsparsity),'_log'];

%%% Step 0 : Initializing the dictionary
disp('on Step 0: estimating UBM noise dictionary and the sample mean supervector')
[W_ubm,W_mean_supervector,c_main] = kd_make_dictionary(nfft,win_len,win_shift,num_of_basis_elements,Wsparsity,Hsparsity,batch_size,datadir,files,dim_of_features);

%%% Step 1: Calculate the 0th and 1st order sufficient statistics & the
%%% variance supervector
disp('on Step 1: calculation of sufficient statistics');
[N_total,F_total,W_covariance_supervector]= kd_calculate_stats_and_cov_vector(num_of_utterances,dim_of_features,num_of_basis_elements,files,Wsparsity,Hsparsity,W_ubm,win_len,win_shift,nfft,W_mean_supervector,datadir,bpre,apre,context_len);
save(['Covarience_supervector_',num2str(num_of_basis_elements),'_',num2str(context_len),'_',num2str(Wsparsity),'_',num2str(Hsparsity)],'W_covariance_supervector');

%%% Step 2: Total Variability Matrix computation
disp('now on step 2: Calculating the Total variability matrix')
T=kd_calculate_TV(W_mean_supervector, W_covariance_supervector, num_of_utterances, F_total, N_total);
save(['T_',num2str(num_of_basis_elements),'_',num2str(context_len),'_',num2str(Wsparsity),'_',num2str(Hsparsity)],'T');

%%% Step 3: Extracting training i-vector -> calculating adapted dictionary
%%% -> calculating the time activation function -> storing it
disp('on Step 3: finding the features for all training utterances')
kd_extract_features_train(files,F_total,N_total,W_mean_supervector,W_covariance_supervector,T,dim_of_features,num_of_basis_elements,bpre,apre,win_len,win_shift,nfft,Wsparsity,Hsparsity,outdir,outdir_log,datadir);

%%%Step 4: Extracting the same for the test set
disp('on Step 4: Calculating the time-activations for the test set')
kd_extract_features_test(bpre,apre,W_ubm,Wsparsity,Hsparsity,dim_of_features,num_of_basis_elements,c_main,W_mean_supervector,W_covariance_supervector,T,outdir,outdir_log,datadir,win_len,win_shift,nfft);

