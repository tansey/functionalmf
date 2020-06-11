% function [u_hpd l_hpd] = calculate_hpd(samples,conf)
%
% Calculates upper (u_hpd) and lower (l_hpd) points of 1D hpd interval
% based on a set of samples and a specified interval length 0<conf<1.

function [u_hpd l_hpd] = calculate_hpd(samples,conf)

num_hpd_samples = floor(length(samples)*conf);

samples = sort(samples,'descend');

u_hpd = samples(1);
l_hpd = samples(num_hpd_samples);

sample_ind = 1;
for u=samples
    samples_below_u = samples < u;
    if sum(samples_below_u) < num_hpd_samples
        break;
    else
        u_hpd_tmp = u;
        l_hpd_tmp = samples(sample_ind + num_hpd_samples - 1);
        
        if u_hpd_tmp - l_hpd_tmp < u_hpd - l_hpd
            u_hpd = u_hpd_tmp;
            l_hpd = l_hpd_tmp;
            
%            length_hpd = u_hpd - l_hpd
        end
        
    end
    sample_ind = sample_ind + 1;
end