function kd_extract_features_test(bpre,apre,W_ubm,Wsparsity,Hsparsity,dim_of_features,num_of_basis_elements,c_main,W_mean_supervector,W_covariance_supervector,T,outdir,datadir,win_len,win_shift,nfft)
    
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
        W_ubm_normalised = W_ubm * diag(1 ./ sqrt(sum(W_ubm.^2, 1)));  % do this bec inside nmf function ,t eh dict is normalised forst , thus op H is wrt to a normalised dict
        c_test_1=((W_ubm_normalised')*sample_mean_test)./diag(W_ubm_normalised'*W_ubm_normalised);
        H = H./c_test_1;
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
%         mkdir([outdir_log, '/', set,'/', noise, '/', spkr_id]);
        dlmwrite([outdir, '/', set,'/', noise, '/', spkr_id, '/', utt_id, '.txt'], H_log', ' ');
%         dlmwrite([outdir_log, '/', set,'/', noise, '/', spkr_id, '/', utt_id, '.txt'], H_log', ' ');
        disp(['Calculated and saved Time activation for Utterance number',num2str(file)])
    end
    disp('All test features extracted!');
end
