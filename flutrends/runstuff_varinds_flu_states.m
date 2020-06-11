%%
    
% Script to run the sampler on the flu data.  Also provides an example of
% the necessary inputs for running the code that analytically marginalizes
% missing data.

load flu_US_states_train;

month_names = {'January','February','March','April','May','June','July','August','September','October','November','December'};

times = datenum(dates);
[years months days] = datevec(dates);

flu = data';

start_dates = zeros(1,size(flu,1));
for ii=1:size(flu,1)
    start_date_ii = 1;
    for tt=1:size(flu,2)
        if isnan(flu(ii,tt))
            start_date_ii = start_date_ii + 1;
        end
    end
    start_dates(ii) = start_date_ii;
end

[q T] = size(flu);

vars = zeros(q,1);
for i=1:q
    vars(i) = var(flu(i,start_dates(i):end));
end

flu = flu./sqrt(max(vars));

y=zeros(q,T);
for i=1:q
    y(i,start_dates(i):end) = flu(i,start_dates(i):end);
end

y = 1.75*y;

tmp = cumsum(sum(y,1));
tmp = find(tmp==0);
if ~isempty(tmp)
    start_time = tmp(end)+1;
    y = y(:,start_time:end-1);
end

inds_y = ones(size(y));
inds_y(find(y==0)) = 0;
inds_y = inds_y > 0;

[p N] = size(y);

x = [1:N]./N;

c = 100;
d = 1;
r = 1e-5;
K = zeros(N);
for ii=1:N
    for jj=1:N
        dist_ii_jj = abs(x(ii)-x(jj));
        K(ii,jj) = d*exp(-c*(dist_ii_jj^2));
    end
end
K = K + diag(r*ones(1,N));
invK = inv(K);
logdetK = 2*sum(log(diag(chol(K))));

prior_params.K.c_prior = 1;
prior_params.K.invK = invK;
prior_params.K.K = K;
prior_params.K.logdetK = logdetK;
prior_params.sig.a_sig = 1;
prior_params.sig.b_sig = 0.1;
prior_params.hypers.a_phi = 1.5;
prior_params.hypers.b_phi = 1.5;
prior_params.hypers.a1 = 10;
prior_params.hypers.a2 = 10;

settings.L = 10;
settings.k = 20;
settings.Niter = 10000;
settings.saveEvery = 100;
settings.storeEvery = 10;
settings.saveMin = 1;
settings.saveDir = 'flu-states';
settings.trial = 1;
settings.init2truth = 0;
settings.sample_K_flag = 3;
settings.latent_mean = 1;
settings.inds_y = inds_y;

BNP_covreg_varinds(y,prior_params,settings,0);


latent_mean = settings.latent_mean;
% inds2impute = settings.inds2impute;
inds2impute = ~settings.inds_y;

sampleEvery = settings.storeEvery;
var_mean = zeros(p,p,N);
var_var = zeros(p,p,N);
var_u = zeros(p,p,N);
var_l = zeros(p,p,N);
mu_mean = zeros(p,N);
mu_var = zeros(p,N);
mu_u = zeros(p,N);
mu_l = zeros(p,N);

cov_true = true_params.cov_true;
mu_true = true_params.mu;

for tt=1:N
    theta_zeta_tt = zeros(p,k,(Niter-Nburn)/sampleEvery);
    var_tt = zeros(p,p,(Niter-Nburn)/sampleEvery);
    mu_tt = zeros(p,(Niter-Nburn)/sampleEvery);
    m = 1;
    for nn=Nburn+1:sampleEvery:Niter
        n = nn+saveEvery-1;
        if rem(n,saveEvery)==0 & n<=Niter
            filename = [saveDir '/BNP_covreg_statsiter' num2str(n) 'trial' num2str(trial) '.mat'];
            load(filename)
            store_count = 1;
        end
        theta_zeta_tt(:,:,m) = Stats(store_count).theta*Stats(store_count).zeta(:,:,tt);
        var_tt(:,:,m) = Stats(store_count).theta*Stats(store_count).zeta(:,:,tt)*Stats(store_count).zeta(:,:,tt)'*Stats(store_count).theta'...
            + diag(1./Stats(store_count).invSig_vec);
        mu_tt(:,m) = Stats(store_count).theta*Stats(store_count).zeta(:,:,tt)*Stats(store_count).psi(:,tt);
        
        m = m + 1;
        store_count = store_count + 1;
    end
    
    var_mean(:,:,tt) = mean(var_tt,3);
    var_var(:,:,tt) = var(var_tt,0,3);
    
    mu_mean(:,tt) = mean(mu_tt,2);
    mu_var(:,tt) = var(mu_tt,0,2);
    
    for pp=1:p
        for jj=pp:p
            [var_u(pp,jj,tt) var_l(pp,jj,tt)] = calculate_hpd(var_tt(pp,jj,:),0.95);
        end
        if latent_mean
            [mu_u(pp,tt) mu_l(pp,tt)] = calculate_hpd(mu_tt(pp,:),0.95);
        end
    end
    
    if ~rem(tt,100)
        display(num2str(tt))
    end
    
end

writematrix(mu_mean, [saveDir '/mu_mean.csv']);

