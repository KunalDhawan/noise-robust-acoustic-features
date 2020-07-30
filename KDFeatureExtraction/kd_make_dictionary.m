function [W_ubm,W_mean_supervector,c_main] = kd_make_dictionary(nfft,win_len,win_shift,num_elems,Wsparsity,Hsparsity,batch_size,datadir,files,dim_of_features)

% NMF  parameters
num_iter=100;
config_github.W_sparsity = Wsparsity;
config_github.H_sparsity = Hsparsity;
config_github.divergence='kl';
config_github.W_init=rand(dim_of_features,num_elems);

% infile = fopen(filelist);
% files = textscan(infile, '%s');
% files = files{:};
files = files(randperm(length(files)));
% fclose(infile);

apre = [1 0.97];
bpre = [1];

total_num_of_frames=0;
sample_mean=zeros(dim_of_features,1);

% Learn dictionary per batch. Initialize dictionary with W from previous batch.
for batch = 1 : floor(length(files) / batch_size)
	display(['Batch ', num2str(batch), ' of ', num2str(floor(length(files) / batch_size))]);
    
	speech = [];
	for k = (batch-1)*batch_size + 1 : batch*batch_size
        [sig, fs] = audioread([datadir, '/', files{k}]);
		sig = filter(bpre,apre,sig);
        speech = [speech; sig];
	end
	speech = speech - mean(speech);
	speech = speech / max(abs(speech));
	speech_spec = spectrogram(speech, round(win_len*fs), round((win_len-win_shift)*fs), nfft, fs, 'yaxis');
    total_num_of_frames=total_num_of_frames + size(abs(speech_spec),2);
    sample_mean=sample_mean+sum(abs(speech_spec),2);
    config_github.maxiter = max(ceil(num_iter / 1.5^(batch-1)),4);
	config_github.W_init = nmf(abs(speech_spec), num_elems, config_github);
end

speech = [];
for k = batch*batch_size + 1 : length(files)
    [sig, fs] = audioread([datadir, '/', files{k}]);
    sig = filter(bpre,apre,sig);
    speech = [speech; sig];
end
speech = speech - mean(speech);
speech = speech / max(abs(speech));
speech_spec = spectrogram(speech, round(win_len*fs), round((win_len-win_shift)*fs), nfft, fs, 'yaxis');
total_num_of_frames=total_num_of_frames + size(abs(speech_spec),2);
sample_mean=sample_mean+sum(abs(speech_spec),2);
config_github.maxiter = max(ceil(num_iter / 1.5^(batch-1)),4);
W_ubm= nmf(abs(speech_spec), num_elems, config_github);
sample_mean=sample_mean/total_num_of_frames;

%W_ubm_old=W_ubm;
c_main=((W_ubm')*sample_mean)./diag(W_ubm'*W_ubm);
W_ubm=W_ubm*diag(c_main);
W_mean_supervector=reshape(W_ubm,[size(W_ubm,1)*size(W_ubm,2),1]);
end



