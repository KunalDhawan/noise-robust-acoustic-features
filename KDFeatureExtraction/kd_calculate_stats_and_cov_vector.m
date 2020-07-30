function [N_total,F_total,W_covariance_supervector]= kd_calculate_stats_and_cov_vector(num_of_utterances,dim_of_features,num_of_basis_elements,files,Wsparsity,Hsparsity,W_ubm,win_len,win_shift,nfft,W_mean_supervector,datadir,bpre,apre,context_len)

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
        spect_mag=abs(spect);
        sample_mean_train=sum(spect_mag,2)/size(spect_mag,2);
        [~, H] = nmf(abs(spect),num_of_basis_elements, config1);
        W_ubm_normalised = W_ubm * diag(1 ./ sqrt(sum(W_ubm.^2, 1)));
        c_train=((W_ubm_normalised')*sample_mean_train)./diag(W_ubm_normalised'*W_ubm_normalised);
        H=H./c_train;
        [N_total(file,: ) , F_total(file,:), cov_num, cov_denum]=kd_calculate_sufficient_statistics( H , abs(spect),cov_num , cov_denum, W_mean_supervector, num_of_basis_elements, dim_of_features );
        disp(['Calculated Sufficient statistics for Utterance number',num2str(file)])
    end

    dlmwrite(['F_total_',num2str(num_of_basis_elements),'_',num2str(context_len),'_',num2str(Wsparsity),'_',num2str(Hsparsity)], F_total, ' ');
    dlmwrite(['N_total_',num2str(num_of_basis_elements),'_',num2str(context_len),'_',num2str(Wsparsity),'_',num2str(Hsparsity)], N_total, ' ');

    %Reconstructing the cov matrix
    cov_denum_reconstructed=repmat(cov_denum,[dim_of_features,1]);
    W_cov_matrix=cov_num./cov_denum_reconstructed;
    W_covariance_supervector=reshape(W_cov_matrix,[num_of_basis_elements * dim_of_features,1]);
end
