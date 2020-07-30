function kd_extract_features_train(files,F_total,N_total,W_mean_supervector,W_covariance_supervector,T,dim_of_features,num_of_basis_elements,bpre,apre,win_len,win_shift,nfft,Wsparsity,Hsparsity,outdir,datadir)

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
%         mkdir([outdir_log, '/', set, '/', spkr_id]);
        dlmwrite([outdir, '/', set, '/', spkr_id, '/', utt_id, '.txt'], H_log', ' ');
%         dlmwrite([outdir_log, '/', set, '/', spkr_id, '/', utt_id, '.txt'], H_log', ' ');
        disp(['Calculated and saved Time activation for Utterance number',num2str(file)])
    end
    disp('All train features extracted!');
    
end
