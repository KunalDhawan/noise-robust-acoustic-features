function H = kd_find_time_activation ( abs_spect,num_of_basis_elements , W_init, Wsparsity, Hsparsity,W_fixed)
    config2.W_sparsity=Wsparsity;
    config2.H_sparsity=Hsparsity;
    config2.divergence='kl';
    config2.W_init = W_init;
    config2.W_fixed = W_fixed;
    config2.maxiter=30;
    [~, H] = nmf(abs_spect,num_of_basis_elements ,config2);
end