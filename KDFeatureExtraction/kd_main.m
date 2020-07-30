% %KD_final_script
% 
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
[W_ubm,sample_mean] = kd_make_dictionary(nfft,win_len,win_shift,num_of_basis_elements,Wsparsity,Hsparsity,batch_size,datadir,filelist,dim_of_features);
W_ubm_old=W_ubm;
c_main=((W_ubm')*sample_mean)./diag(W_ubm'*W_ubm);
W_ubm=W_ubm*diag(c_main);
W_mean_supervector=reshape(W_ubm,[size(W_ubm,1)*size(W_ubm,2),1]);
save(['W_ubm_',num2str(num_of_basis_elements),'_',num2str(context_len),'_',num2str(Wsparsity),'_',num2str(Hsparsity)],'W_ubm');
save(['Mean_supervector_',num2str(num_of_basis_elements),'_',num2str(context_len),'_',num2str(Wsparsity),'_',num2str(Hsparsity)],'W_mean_supervector');

%%% Step 1: Calculate the 0th and 1st order sufficient statistics & the
%%% variance supervector
disp('on Step 1: calculation of sufficient statistics');

F_total=zeros(num_of_utterances,num_of_basis_elements*dim_of_features);
N_total=zeros(num_of_utterances,num_of_basis_elements);

cov_num=zeros(dim_of_features,num_of_basis_elements);
cov_denum=zeros(1,num_of_basis_elements);
config1.divergence='kl';
config1.W_sparsity=Wsparsity;
config1.W_init = W_ubm;
config1.W_fixed = true;
config1.H_sparsity = Hsparsity;

for file=1:length(files)
    [sig, fs] = audioread([datadir, '/', files{file}]);
    sig = filter(bpre,apre,sig);
    sig = sig - mean(sig);
    sig = sig / max(abs(sig));
    spect=spectrogram(sig, round(win_len*fs), round((win_len-win_shift)*fs), nfft, fs, 'yaxis');
    [~, H] = nmf(abs(spect),num_of_basis_elements, config1);
    H=H./c_main;
    [N_total(file,: ) , F_total(file,:), cov_num, cov_denum]=kd_calculate_sufficient_statistics( H , abs(spect),cov_num , cov_denum, W_mean_supervector, num_of_basis_elements, dim_of_features );
    disp(['Calculated Sufficient statistics for Utterance number',num2str(file)])
end

dlmwrite(['F_total_',num2str(num_of_basis_elements),'_',num2str(context_len),'_',num2str(Wsparsity),'_',num2str(Hsparsity)], F_total, ' ');
dlmwrite(['N_total_',num2str(num_of_basis_elements),'_',num2str(context_len),'_',num2str(Wsparsity),'_',num2str(Hsparsity)], N_total, ' ');

%Reconstructing the cov matrix
cov_denum_reconstructed=repmat(cov_denum,[dim_of_features,1]);
W_cov_matrix=cov_num./cov_denum_reconstructed;
W_covariance_supervector=reshape(W_cov_matrix,[num_of_basis_elements * dim_of_features,1]);
save(['Covarience_supervector_',num2str(num_of_basis_elements),'_',num2str(context_len),'_',num2str(Wsparsity),'_',num2str(Hsparsity)],'W_covariance_supervector');

%%% Step 2: Total Variability Matrix computation

disp('now on step 2: Calculating the Total variability matrix')
T=kd_calculate_TV(W_mean_supervector, W_covariance_supervector, num_of_utterances, F_total, N_total);
save(['T_',num2str(num_of_basis_elements),'_',num2str(context_len),'_',num2str(Wsparsity),'_',num2str(Hsparsity)],'T');

%%% Step 3: Extracting training i-vector -> calculating adapted dictionary
%%% -> calculating the time activation function -> storing it
disp('on Step 3: finding the features for all training utterances')

set='train_multi';

parfor file=1:length(files)
    F=F_total(file,:);
    N=N_total(file,:);
    w = estimate_y_and_v(F, N, 0, W_mean_supervector, W_covariance_supervector, 0, T, 0, 0, 0, 0, 1);
    w_updated=W_mean_supervector + T'*w';
    w_updated(w_updated<0)=eps;
    w_updated_reconstructed=reshape(w_updated,[dim_of_features,num_of_basis_elements]);
    
    [filepath, utt_id] = fileparts(files{file});
    spkr_id = utt_id(1:3);
    [sig, fs] = audioread([datadir, '/', files{file}]);
    sig = filter(bpre,apre,sig);
    sig = sig - mean(sig);
    sig = sig / max(abs(sig));
    spect=spectrogram(sig, round(win_len*fs), round((win_len-win_shift)*fs), nfft, fs, 'yaxis');
    spect_mag=abs(spect);
    sample_mean_train=sum(spect_mag,2)/size(spect_mag,2);   
    W_init = w_updated_reconstructed;
    W_fixed = true;
    H = kd_find_time_activation ( abs(spect),num_of_basis_elements , W_init, Wsparsity, Hsparsity,W_fixed);
    if ( sum(sum(isnan(H))) ~= 0)
        keyboard;
    end    
    H(H==0)=eps;
    w_updated_reconstructed = w_updated_reconstructed * diag(1 ./ sqrt(sum(w_updated_reconstructed.^2, 1)));
    c_train=((w_updated_reconstructed')*sample_mean_train)./diag(w_updated_reconstructed'*w_updated_reconstructed);
    H=H./c_train;
    H_log=log10(H);
    
    mkdir([outdir, '/', set, '/', spkr_id]);
    mkdir([outdir_log, '/', set, '/', spkr_id]);
    dlmwrite([outdir, '/', set, '/', spkr_id, '/', utt_id, '.txt'], H', ' ');
    dlmwrite([outdir_log, '/', set, '/', spkr_id, '/', utt_id, '.txt'], H_log', ' ');
    disp(['Calculated and saved Time activation for Utterance number',num2str(file)])
end
disp('All train features extracted!');
%%%Step 4: Extracting the same for the test set
disp('Calculating the time-activations for the test set')

set= 'test_0330';
kdID=fopen('/easyshare/kunal/aurora4/lists/my_lists/test_0330_16k.list');
files = textscan(kdID, '%s');
files = files{:};
fclose(kdID);

parfor file=1:length(files)
    [sig, fs] = audioread([datadir, '/', files{file}]);
    sig = filter(bpre,apre,sig);
    sig = sig - mean(sig);
    sig = sig / max(abs(sig));
    spect=spectrogram(sig, round(win_len*fs), round((win_len-win_shift)*fs), nfft, fs, 'yaxis');
    spect_mag=abs(spect);
    sample_mean_test=sum(spect_mag,2)/size(spect_mag,2);
    W_init = W_ubm;
    W_fixed = true;
    H = kd_find_time_activation ( abs(spect),num_of_basis_elements , W_init, Wsparsity, Hsparsity,W_fixed);
    H = H./c_main;
    [N , F]=kd_calculate_sufficient_statistics_only( H , abs(spect) );
    w = estimate_y_and_v(F, N, 0, W_mean_supervector, W_covariance_supervector, 0, T, 0, 0, 0, 0, 1);
    w_updated=W_mean_supervector + T'*w';

    w_updated(w_updated<0)=eps;
    w_updated_reconstructed=reshape(w_updated,[dim_of_features,num_of_basis_elements]);
   
    [filepath, utt_id] = fileparts(files{file});
    filepath_parts = strsplit(filepath, '/');
    noise = filepath_parts{2};
    spkr_id = utt_id(1:3);
    W_init = w_updated_reconstructed;
    W_fixed = true;
    H = kd_find_time_activation ( abs(spect),num_of_basis_elements , W_init, Wsparsity, Hsparsity,W_fixed);
    if ( sum(sum(isnan(H))) ~= 0)
        keyboard;
    end 
    H(H==0)=eps;
    w_updated_reconstructed = w_updated_reconstructed * diag(1 ./ sqrt(sum(w_updated_reconstructed.^2, 1)));
    c_test=((w_updated_reconstructed')*sample_mean_test)./diag(w_updated_reconstructed'*w_updated_reconstructed);
    H=H./c_test;
    H_log=log10(H);
    mkdir([outdir, '/', set,'/', noise, '/', spkr_id]);
    mkdir([outdir_log, '/', set,'/', noise, '/', spkr_id]);
    dlmwrite([outdir, '/', set,'/', noise, '/', spkr_id, '/', utt_id, '.txt'], H', ' ');
    dlmwrite([outdir_log, '/', set,'/', noise, '/', spkr_id, '/', utt_id, '.txt'], H_log', ' ');
    disp(['Calculated and saved Time activation for Utterance number',num2str(file)])
end
disp('All test features extracted!');

