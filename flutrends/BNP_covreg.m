% BNP_covreg(y,prior_params,settings,restart,true_params)
%
% Method for producing samples of predictor-dependent mean and covariance.
% If there are missing values, this script allows for imputing them during
% the sampling.  See BNP_covreg_varinds for a sampler that analytically
% marginalizes the missing values.
%
% In this model:
% y_i = \Theta\xi(x_i)\eta_i + \epsilon_i
% \eta_i = \psi(x_i) + \xi_i
% \epsilon_i \sim N(0,\Sigma_0),    \xi_i \sim N(0,I).
%
% If one wishes to assume zero latent mean \mu(x) = 0, the elements \psi(x)
% are assumed to be zero instead of GPs.
%
% Required Inputs:
% y - p x n matrix of data
% prior_params - hyperparameters for priors (see example)
% settings - settings for number of itereations, saved statistics, etc.
%
% Optional Inputs:
% restart - binary variable indicating whether or not we are continuing
% on using a previous chain of Gibbs samples
% true_params - structure containing the true mean and covariance used only
% for generating plots during the sampling


function BNP_covreg(y,prior_params,settings,restart,true_params)

% Settings for adaptation, if desired.  Settings of T0 = 1 and Tf = 1
% indicate no adaptation.
T0 = 1;
Tf = 1;
n_adapt = 1000;
temp = T0;

% Determine whether or not to use the quasi empirical Bayes method outlined
% in the paper to initialize the sampler:
empBayes = 1;

[p N] = size(y);

if exist('true_params','var')
    cov_true_diag = zeros(p,N);
    for tt=1:N
        cov_true_diag(:,tt) = diag(true_params.cov_true(:,:,tt));
    end
end

if ~exist('restart','var')
    restart = 0;
end

% Impute any missing values:
y = init_y(y,settings,true_params);

% K represents the correlation matrix of the latent GPs.  There are three
% options for how to handle K during sampling, as determined below:
sample_K_flag = settings.sample_K_flag; % (1) sample K marginalizing zeta, (2) sample K conditioning on zeta, (3) set K to fixed value

% Indicate whether or not we wish to model a latent mean \mu(x):
latent_mean = settings.latent_mean;

k = settings.k;  % dimension of latent factors
L = settings.L;  % number of dictionary elements = L*k:
Niter = settings.Niter;   % number of Gibbs iterations
trial = settings.trial;   % label for MCMC chain

if ~restart
    
    % Initialize structure for storing samples:
    Stats(1:settings.saveEvery/settings.storeEvery) = struct('zeta',zeros(L,k,N),'psi',zeros(k,N),'invSig_vec',zeros(1,p),...
        'theta',zeros(p,L),'eta',zeros(k,N),'phi',zeros(p,L),'tau',zeros(1,L),'K_ind',0,'y_heldout',zeros(1,sum(inds2impute(:))));
    store_counter = 1;
    
    % Sample hyperparams from prior:
    delta = zeros(1,L);
    delta(1) = gamrnd(prior_params.hypers.a1,1);
    delta(2:L) = gamrnd(prior_params.hypers.a2*ones(1,L-1),1);
    tau = exp(cumsum(log(delta)));
    phi = gamrnd(prior_params.hypers.a_phi*ones(p,L),1) / prior_params.hypers.b_phi;
    
    % Sample theta, eta, and Sigma initially as prior draws or using the quasi
    % empirical Bayes method outlined in the paper:
    theta = zeros(p,L);
    for pp=1:p
        theta(pp,:) = chol(diag(1./(phi(pp,:).*tau)))'*randn(L,1);
    end
    
    invSig_vec = gamrnd(prior_params.sig.a_sig*ones(1,p),1) / prior_params.sig.b_sig;
    
    psi = zeros(k,N);
    if empBayes
        xi = sample_xi_init(y,invSig_vec,psi,temp);
    else
        xi = randn(k,N);
    end
    eta = psi + xi;
    
    
    % Sample initial GP cov K and GP latent functions zeta_i:
    if sample_K_flag==1 || sample_K_flag==2
        Pk = cumsum(prior_params.K.c_prior);
        K_ind = 1 + sum(Pk(end)*rand(1) > Pk);
    else
        K_ind = 1;
    end
    % Sample zeta_i using initialization scheme based on data and other
    % sampled params:
    
    if empBayes
        
        zeta = zeros(L,k,N);
        for ii=1:10
            [zeta Sig_est] = initialize_zeta(zeta,y,theta,invSig_vec);
            
            zeta = sample_zeta(y,theta,eta,invSig_vec,zeta,prior_params.K.invK(:,:,K_ind),temp);
            theta = sample_theta(y,eta,invSig_vec,zeta,phi,tau,temp);
            invSig_vec = sample_sig(y,theta,eta,zeta,prior_params.sig,temp);
        end
        
    else
        zeta = zeros(L,k,N);
        cholK = chol(prior_params.K.K);
        for ll=1:L
            for kk=1:k
                zeta(ll,kk,:) = cholK'*randn(N,1);
            end
        end
        zeta = sample_zeta(y,theta,eta,invSig_vec,zeta,prior_params.K.invK(:,:,K_ind),temp);
    end
    
    
    if latent_mean
        psi = sample_psi_margxi(y,theta,invSig_vec,zeta,psi,prior_params.K.invK(:,:,K_ind),temp);
    else
        psi = zeros(size(xi));
    end
    % Sample latent factors given the data, weightings matrix, latent GP
    % functions, and noise parameters.
    xi = sample_xi(y,theta,invSig_vec,zeta,psi,temp);
    
    eta = psi + xi;
    
    % Impute any missing y's:
    if ~isempty(inds2impute)
        y = sample_y(y,theta,invSig_vec,zeta,psi,inds2impute);
    end
    
    % Create directory for saving statistics if it does not exist:
    if ~exist(settings.saveDir,'file')
        mkdir(settings.saveDir);
    end
    
    % Save initial statistics and settings for this trial/chain:
    if isfield(settings,'filename')
        settings_filename = strcat(settings.saveDir,'/',settings.filename,'_info4trial',num2str(trial));    % create filename for current iteration
        init_stats_filename = strcat(settings.saveDir,'/',settings.filename,'initialStats_trial',num2str(trial));    % create filename for current iteration
    else
        settings_filename = strcat(settings.saveDir,'/info4trial',num2str(trial));    % create filename for current iteration
        init_stats_filename = strcat(settings.saveDir,'/initialStats_trial',num2str(trial));    % create filename for current iteration
    end
    if nargin>3
        save(settings_filename,'y','settings','prior_params','true_params') % save current statistics
    else
        save(settings_filename,'y','settings','prior_params') % save current statistics
    end
    save(init_stats_filename,'zeta','psi','eta','theta','invSig_vec','phi','tau','K_ind') % save current statistics
    
    num_iters = 1;
    
    nstart = 1;
    
else
    
    % Load last saved sample from the Gibbs chain to restart the sampler:
    load([settings.saveDir,'/BNP_covreg_stats','iter',num2str(settings.lastIter),'trial',num2str(settings.trial)]);
    
    theta = Stats(end).theta;
    eta = Stats(end).eta;
    zeta = Stats(end).zeta;
    phi = Stats(end).phi;
    tau = Stats(end).tau;
    psi = Stats(end).psi;
    
    nstart = settings.lastIter + 1;
    num_iters = 1;
    store_counter = 1;
end

for nn=nstart:Niter
    
    % Sample latent factor model additive Gaussian noise covariance
    % (diagonal matrix) given the data, weightings matrix, latent GP
    % functions, and latent factors:
    invSig_vec = sample_sig(y,theta,eta,zeta,prior_params.sig,temp);
    
    % Sample weightings matrix hyperparameters given weightings matrix:
    [phi tau] = sample_hypers(theta,phi,tau,prior_params.hypers);
    
    % Sample weightings matrix given the data, latent factors, latent GP
    % functions, noise parameters, and hyperparameters:
    theta = sample_theta(y,eta,invSig_vec,zeta,phi,tau,temp);
    
    % Sample K based on the p x N dimensional Gaussian posterior
    % p(K | y, theta, eta, Sigma_0) having marginalized the latent GP
    % functions zeta.  If p x N is too large, sample the GP cov matrix K
    % given the latent GP functions zeta p(K | zeta) (yielding K cond ind
    % of everything else, though leading to much slower mixing rates):
    if sample_K_flag == 1
        K_ind = sample_K_marg_zeta(y,theta,eta,invSig_vec,prior_params.K,K_ind);
    elseif sample_K_flag == 2
        K_ind = sample_K_cond_zeta(zeta,prior_params.K);
    else
        K_ind = 1;  % set K to fixed value (i.e., do not resample hyperparameters)
    end
    
    % The following represents a block sampling of \psi and \xi.
    % If modeling a non-zero latent mean \mu(x), sample the latent mean
    % GPs \psi(x) marginalizing \xi.  Otherwise, set these elements to
    % zeros:
    if latent_mean
        psi = sample_psi_margxi(y,theta,invSig_vec,zeta,psi,prior_params.K.invK(:,:,K_ind),temp);
    else
        psi = zeros(size(xi));
    end
    % Sample latent factors given the data, weightings matrix, latent GP
    % functions, and noise parameters.
    xi = sample_xi(y,theta,invSig_vec,zeta,psi,temp);
    
    eta = psi + xi;
    
    % Sample latent GP functions zeta_i given the data, weightings matrix,
    % latent factors, noise params, and GP cov matrix (hyperparameter):
    
    for ii=1:num_iters
        zeta = sample_zeta(y,theta,eta,invSig_vec,zeta,prior_params.K.invK(:,:,K_ind),temp);
    end
    
    % Impute any missing y's:
    if ~isempty(inds2impute)
        y = sample_y(y,theta,invSig_vec,zeta,psi,inds2impute);
    end
    
    num_iters = 1;
    
    if nn<=n_adapt
        temp = T0*((Tf/T0)^(nn/n_adapt));
    end
    
    % If the current Gibbs iteration is a multiple of the frequency at
    % which samples are stored, add current param samples to Stats struct:
    if rem(nn,settings.storeEvery)==0 & nn>=settings.saveMin
        Stats(store_counter).zeta = zeta;
        Stats(store_counter).eta = eta;
        Stats(store_counter).theta = theta;
        Stats(store_counter).invSig_vec = invSig_vec;
        Stats(store_counter).psi = psi;
        Stats(store_counter).phi = phi;
        Stats(store_counter).tau = tau;
        Stats(store_counter).K_ind = K_ind;
        
        y_tmp = y(inds2impute);
        Stats(store_counter).y_heldout = y_tmp(:);
        store_counter = store_counter + 1;
    end
    
    % If the current Gibbs iteration is a multiple of the frequency at
    % which statistics are saved, save Stats structure to hard drive:
    if rem(nn,settings.saveEvery)==0
        
        % Create filename for saving current Stats struct:
        if isfield(settings,'filename')
            filename = strcat(settings.saveDir,'/',settings.filename,'iter',num2str(nn),'trial',num2str(settings.trial));    % create filename for current iteration
        else
            filename = strcat(settings.saveDir,'/BNP_covreg_stats','iter',num2str(nn),'trial',num2str(settings.trial));    % create filename for current iteration
        end
        
        % Save stats to specified directory:
        save(filename,'Stats')
        
        % Reset S counter variables:
        store_counter = 1;
        
    end
    
    % Plot some stats:
    if ~rem(nn,100)
        display(['Iter: ' num2str(nn) ', K_ind: ' num2str(K_ind)])
        if exist('true_params','var')
            cov_est = zeros(p,N);
            for tt=1:N
                cov_est(:,tt) = diag(theta*zeta(:,:,tt)*zeta(:,:,tt)'*theta' + diag(1./invSig_vec));
            end
            plot(cov_true_diag','LineWidth',2); hold on; plot(cov_est','--','LineWidth',2); hold off; drawnow;
        end
    end
    
end

return;

function zeta = sample_zeta(y,theta,eta,invSig_vec,zeta,invK,temp)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% The posterior of all latent GPs can be analytically computed in      %%
%%% closed form.  Specifically, it is an NxkxL dimensional Gaussian.     %%
%%% However, for most problems, it is infeasible to sample directly from %%
%%% this joint posterior because of the dimensionality of the Gaussian   %%
%%% parameterization.  Below is the code for sampling from this joint    %%
%%% posterior for reference:                                             %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [p N] = size(y);
% L = size(theta,2);
% k = size(eta,1);
%
% invK = prior_params.invK;
%
% AinvSig = zeros(N*k*L,N*p);
% AinvSigA = zeros(N*k*L);
% AinvSig = sparse(AinvSig);
% AinvSigA = sparse(AinvSigA);
% for nn=1:N
%     tmp1 = kron(theta,eta(:,nn)');
%     tmp2 = tmp1'*invSig;
%     AinvSig((nn-1)*L*k+1:nn*L*k,(nn-1)*p+1:nn*p) = tmp2;
%     AinvSigA((nn-1)*L*k+1:nn*L*k,(nn-1)*L*k+1:nn*L*k) = tmp2*tmp1;
% end
%
% invbigK = kron(invK,eye(L*k));  % inv(kron(K,eye(L*k)));
% invbigK = sparse(invbigK);
%
% Sig = (invbigK + AinvSigA) \ eye(N*k*L);
% m = Sig*(AinvSig*y(:));
% zeta_vec = m + chol(Sig)'*randn(N*k*L,1);
%
% zeta = zeros(L,k,N);
% for ii=1:N
%     zeta_tmp = zeta_vec((ii-1)*k*L + 1: ii*k*L);
%     zeta(:,:,ii) = reshape(zeta_tmp,[k L])';
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Instead, we sample the latent GP functions as follows.  For          %%
%%% initialization we sequentially walk through each row sampling        %%
%%% zeta_{ll,kk} assuming zeta_{ll+1:L,unsampled kk for row ll}=0.       %%
%%% We move in this order since in expectation the importance of each    %%
%%% zeta_{ll,kk} decreases with increasing ll due to the sparsity        %%
%%% structure of the weightings matrix theta.  We then reloop through    %%
%%% sampling the zeta_{ll,kk} multiple times in order to improve the     %%
%%% mixing rate given the other currently sampled params.  The other     %%
%%% calls to resampling zeta_{ll,kk} operates in exactly the same way,   %%
%%% but based on the past sample of zeta.                                %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[p L] = size(theta);
[k N] = size(eta);

% We derive the sequential sampling of the zeta_{ll,kk} by reformulating
% the full regression problem as one that is solely in terms of
% zeta_{ll,kk} having conditioned on the other latent GP functions:
%
% y_i = eta(i,m)*theta(:,ll)*zeta_{ll,kk}(x_i) + tilde(eps)_i,
%

% Initialize the structure for holding the conditional mean for additive
% Gaussian noise term tilde(eps)_i and add values based on previous zeta:
mu_tot = zeros(p,N);
for nn=1:N
    mu_tot(:,nn) = theta*zeta(:,:,nn)*eta(:,nn);
end

for ll=1:L  % walk through each row of zeta matrix sequentially
    theta_ll = theta(:,ll);
    for kk=randperm(k);  % create random ordering for kk in sampling zeta_{ll,kk}
        eta_kk = eta(kk,:);
        zeta_ll_kk = squeeze(zeta(ll,kk,:))';
        mu_tot = mu_tot - theta_ll(:,ones(1,N)).*eta_kk(ones(p,1),:).*zeta_ll_kk(ones(p,1),:);
        
        % Using standard Gaussian identities, form posterior of
        % zeta_{ll,kk} using information form of Gaussian prior and likelihood:
        A_lk_invSig_A_lk = diag((eta(kk,:).^2)*((theta(:,ll).^2)'*invSig_vec'));
        theta_tmp = theta(:,ll)'.*invSig_vec;
        ytilde = y - mu_tot;  % normalize data by subtracting mean of tilde(eps)
        theta_lk = eta(kk,:)'.*(theta_tmp*ytilde)'; % theta_lk = eta(kk,:)'.*diag(theta_tmp(ones(1,N),:)*ytilde);
        
        % Transform information parameters:
        cholSig_lk_trans = chol(invK + A_lk_invSig_A_lk) \ eye(N);  % Sig_lk = inv(invK + A_lk_invSig_A_lk);
        m_lk = cholSig_lk_trans*(cholSig_lk_trans'*theta_lk);  % m_lk = Sig_lk*theta_lk;
        
        % Sample zeta_{ll,kk} from posterior Gaussian:
        zeta(ll,kk,:) = m_lk + temp*cholSig_lk_trans*randn(N,1);  % zeta(ll,kk,:) = m_lk + chol(Sig_lk)'*randn(N,1);
        
        zeta_ll_kk = squeeze(zeta(ll,kk,:))';
        mu_tot = mu_tot + theta_ll(:,ones(1,N)).*eta_kk(ones(p,1),:).*zeta_ll_kk(ones(p,1),:);
    end
end


return;

function psi = sample_psi_margxi(y,theta,invSig_vec,zeta,psi,invK,temp)

[p L] = size(theta);
[k N] = size(psi);
Sigma_0 = diag(1./invSig_vec);

% We derive the sequential sampling of the zeta_{ll,kk} by reformulating
% the full regression problem as one that is solely in terms of
% zeta_{ll,kk} having conditioned on the other latent GP functions:
%
% y_i = eta(i,m)*theta(:,ll)*zeta_{ll,kk}(x_i) + tilde(eps)_i,
%

% Initialize the structure for holding the conditional mean for additive
% Gaussian noise term tilde(eps)_i and add values based on previous zeta:
mu_tot = zeros(p,N);
Omega = zeros(p,k,N);
invOmegaOmegaSigma0 = zeros(p,p,N);
for nn=1:N
    Omega(:,:,nn) = theta*zeta(:,:,nn);
    temp = (Omega(:,:,nn)*Omega(:,:,nn)' + Sigma_0) \ eye(N);
    OmegaInvOmegaOmegaSigma0(:,:,nn) = Omega(:,:,nn)'*temp;
    mu_tot(:,nn) = Omega(:,:,nn)*psi(:,nn);
end

if sum(sum(mu_tot))==0 % if this is a call to initialize psi
    numTotIters = 100;
else
    numTotIters = 5;
end

for numIter = 1:numTotIters
    
    for kk=randperm(k);  % create random ordering for kk in sampling zeta_{ll,kk}
        
        Omega_kk = squeeze(Omega(:,kk,:));
        psi_kk = psi(kk,:);
        mu_tot = mu_tot - Omega_kk.*psi_kk(ones(p,1),:);
        
        theta_k = diag(squeeze(OmegaInvOmegaOmegaSigma0(kk,:,:))'*(y-mu_tot));
        Ak_invSig_Ak = diag(squeeze(OmegaInvOmegaOmegaSigma0(kk,:,:))'*Omega_kk);
        
        cholSig_k_trans = chol(invK + diag(Ak_invSig_Ak)) \ eye(N);
        
        % Transform information parameters:
        m_k = cholSig_k_trans*(cholSig_k_trans'*theta_k);  % m_lk = Sig_lk*theta_lk;
        
        % Sample zeta_{ll,kk} from posterior Gaussian:
        psi(kk,:) = m_k + temp*cholSig_k_trans*randn(N,1);  % zeta(ll,kk,:) = m_lk + chol(Sig_lk)'*randn(N,1);
        
        psi_kk = psi(kk,:);
        mu_tot = mu_tot + Omega_kk.*psi_kk(ones(p,1),:);
    end
    
end

return;

function xi = sample_xi(y,theta,invSig_vec,zeta,psi,temp)

% Sample latent factors eta_i using standard Gaussian identities based on
% the fact that:
%
% y_i = (theta*zeta)*eta_i + eps_i,   eps_i \sim N(0,Sigma_0),   eta_i \sim N(0,I)
%
% and using the information form of the Gaussian likelihood and prior.

[p N] = size(y);
[L k] = size(zeta(:,:,1));

invSigMat = invSig_vec(ones(k,1),:);

xi = zeros(k,N);
for nn=1:N
    theta_zeta_n = theta*zeta(:,:,nn);
    y_tilde_n = y(:,nn)-theta_zeta_n*psi(:,nn);
    zeta_theta_invSig = theta_zeta_n'.*invSigMat;  % zeta_theta_invSig = zeta(:,:,nn)'*(theta'*invSig);
    
    cholSig_xi_n_trans = chol(eye(k) + zeta_theta_invSig*theta_zeta_n) \ eye(k);  % Sig_eta_n = inv(eye(k) + zeta_theta_invSig*theta_zeta_nn);
    m_xi_n = cholSig_xi_n_trans*(cholSig_xi_n_trans'*(zeta_theta_invSig*y_tilde_n));  % m_eta_n = Sig_eta_n*(zeta_theta_invSig*y(:,nn));
    
    xi(:,nn) = m_xi_n + temp*cholSig_xi_n_trans*randn(k,1); % eta(:,nn) = m_eta_n + chol(Sig_eta_n)'*randn(k,1);
end

return;

function theta = sample_theta(y,eta,invSig_vec,zeta,phi,tau,temp)

[p N] = size(y);
L = size(zeta,1);
theta = zeros(p,L);

eta_tilde = zeros(L,N);
for nn=1:N
    eta_tilde(:,nn) = zeta(:,:,nn)*eta(:,nn);
end
eta_tilde = eta_tilde';

for pp=1:p
    %%% Possibly numerically more stable:
    % invThetaCovPrior = diag(1./(phi(pp,:).*tau));
    % ThetaCov = invThetaCovPrior - invThetaCovPrior*inv(inv(eta_tilde'*eta_tilde)+invThetaCovPrior)*invThetaCovPrior;
    chol_Sig_theta_p_trans = chol(diag(phi(pp,:).*tau) + invSig_vec(pp)*(eta_tilde'*eta_tilde)) \ eye(L);
    m_theta_p = invSig_vec(pp)*(chol_Sig_theta_p_trans*chol_Sig_theta_p_trans')*(eta_tilde'*y(pp,:)');
    theta(pp,:) = m_theta_p + temp*chol_Sig_theta_p_trans*randn(L,1);
end

return;


function invSig_vec = sample_sig(y,theta,eta,zeta,prior_params,temp)

[p N] = size(y);

a_sig = prior_params.a_sig;
b_sig = prior_params.b_sig;

invSig_vec = zeros(1,p);
for pp = 1:p
    sq_err = 0;
    for nn=1:N
        sq_err = sq_err + (y(pp,nn) - theta(pp,:)*zeta(:,:,nn)*eta(:,nn))^2;
    end
    a_temp = 1 + (a_sig + 0.5*N - 1)/temp;
    b_temp = (b_sig + 0.5*sq_err)/temp;
    invSig_vec(pp) = gamrnd(a_temp,1) / b_temp; %invSig_vec(pp) = gamrnd(a_sig + 0.5*N,1) / (b_sig + 0.5*sq_err);
end

return;

function [phi tau] = sample_hypers(theta,phi,tau,prior_params)

[p L] = size(theta);

a1 = prior_params.a1;
a2 = prior_params.a2;
a_phi = prior_params.a_phi;
b_phi = prior_params.b_phi;

a = [a1 a2*ones(1,L-1)];
delta = exp([log(tau(1)) diff(log(tau))]);

for numIter = 1:50
    
    phi = gamrnd((a_phi + 0.5)*ones(p,L),1) ./ (b_phi + 0.5*tau(ones(1,p),:).*(theta.^2));
    
    sum_phi_theta = sum(phi.*(theta.^2),1);
    for hh=1:L
        tau_hh = exp(cumsum(log(delta))).*[zeros(1,hh-1) ones(1,L-hh+1)./delta(hh)];
        delta(hh) = gamrnd(a(hh) + 0.5*p*(L-hh+1),1) ./ (1 + 0.5*sum(tau_hh.*sum_phi_theta));
    end
    
    tau = exp(cumsum(log(delta)));
    
end

return;


function K_ind = sample_K_marg_zeta(y,theta,eta,invSig_vec,prior_params,K_ind)

[k N] = size(eta);
[p L] = size(theta);
c_prior = prior_params.c_prior;
K = prior_params.K;

grid_size = length(c_prior);

Pk = -Inf*ones(1,grid_size);
Pk_tmp = -Inf*ones(1,grid_size);

% For computational reasons, one can restrict to examining just a local
% neighborhood of the current length scale parameter, but not exact:
nbhd = Inf; %ceil(grid_size/100);
nbhd_vec = [max(K_ind-nbhd,1):min(K_ind+nbhd,grid_size)];

tmp_mat_init = diag(repmat(1./invSig_vec,1,N));

if p < sqrt(N)
    
    for cc=nbhd_vec
        Kc = K(:,:,cc);
        tmp_mat = tmp_mat_init;
        for ll=1:L
            theta_ll = theta(:,ll);
            for kk=1:k
                eta_kk = eta(kk,:);
                
                tmp_mat = tmp_mat + kron((eta_kk*eta_kk').*Kc,theta_ll*theta_ll');
            end
        end
        
        Pk(cc) = normpdfln(y(:),zeros(p*N,1),tmp_mat);
    end
    
else
    
    for cc=nbhd_vec
        Kc = K(:,:,cc);
        Kc = sparse(Kc);
        tmp_mat = tmp_mat_init;
        for ll=1:L
            theta_ll = theta(:,ll);
            for kk=1:k
                eta_kk = eta(kk,:);
                eta_mat_kk = diag(eta_kk);
                eta_theta_ll_kk = kron(eta_mat_kk,theta_ll);
                
                tmp_mat = tmp_mat + eta_theta_ll_kk*Kc*eta_theta_ll_kk';
            end
        end

        Pk(cc) = normpdfln(y(:),zeros(p*N,1),tmp_mat);
    end
    
end

nbhd_mask = zeros(1,length(prior_params.c_prior));
nbhd_mask(nbhd_vec) = 1;
Pk = Pk + log(prior_params.c_prior).*nbhd_mask;

Pk = cumsum(exp(Pk-max(Pk)));
K_ind = 1 + sum(Pk(end)*rand(1) > Pk);

return;


function K_ind = sample_K_cond_zeta(zeta,prior_params)

[L k N] = size(zeta);
c_prior = prior_params.c_prior;
invK = prior_params.invK;

zeta_tmp = reshape(zeta,[L*k N])';

Pk = zeros(1,length(c_prior));
for ii=1:length(c_prior)
    Pk(ii) = - sum(diag(zeta_tmp'*(invK(:,:,ii)*zeta_tmp)));
end
Pk = Pk - 0.5*(L*k)*prior_params.logdetK + log(prior_params.c_prior);
Pk = cumsum(exp(Pk-max(Pk)));
K_ind = 1 + sum(Pk(end)*rand(1) > Pk);

return;

function y = sample_y(y,theta,invSig_vec,zeta,psi,inds2impute)

% inds2impute:
% Binary matrix with elements (j,n) indicating whether the n^{th} obs
% vector is missing component j.

[p N] = size(y);

inds_vec = [1:p];

for nn=1:N
    
    Sigma_nn = theta*zeta(:,:,nn)*zeta(:,:,nn)'*theta' + diag(1./invSig_vec);
    mu_nn = theta*zeta(:,:,nn)*psi(:,nn);
    
    inds2impute_nn = inds_vec(inds2impute(:,nn));  J = length(inds2impute_nn);
    reg_inds_nn = setdiff(inds_vec,inds2impute_nn);
    
    ordered_inds = [inds2impute_nn reg_inds_nn];
    
    % reorder cov and mean:
    ordered_cov_rows = zeros(p);
    ordered_mean = zeros(p,1);
    for jj=1:p
        ordered_cov_rows(jj,:) = Sigma_nn(ordered_inds(jj),:);
        ordered_mean(jj) = mu_nn(ordered_inds(jj));
    end
    ordered_cov = zeros(size(ordered_cov_rows));
    for jj=1:p
        ordered_cov(:,jj) = ordered_cov_rows(:,ordered_inds(jj));
    end
    
    Sig11 = ordered_cov(1:J,1:J);
    Sig12 = ordered_cov(1:J, J+1:p);
    Sig12invSig22 = Sig12 / ordered_cov(J+1:p, J+1:p);
    
    mu1 = ordered_mean(1:J);
    mu2 = ordered_mean(J+1:p);
    
    reg_mean = mu1 + Sig12invSig22*(y(reg_inds_nn,nn) - mu2);
    reg_cov = Sig11 - Sig12invSig22*Sig12';
    
    y(inds2impute_nn,nn) = reg_mean + chol(reg_cov)'*randn(J,1);
    
end

return;

function y = init_y(y,settings,true_params)

inds2impute = settings.inds2impute;
[p N] = size(y);
y(inds2impute) = 0;

x = linspace(-1,1,5); mu = 0; sig = 1;
tmp = normpdf(x,mu,sig);

y_conved = y;
for pp=1:p
    y_conved(pp,:) = conv(y(pp,:),tmp,'same');
end

y(inds2impute) = y_conved(inds2impute);

return;

function [zeta Sig_est] = initialize_zeta(zeta,y,theta,invSig_vec)

[p N] = size(y);
[L k tmp] = size(zeta);

Sig_mat = diag(1./invSig_vec);

Sig_est = zeros(p,N);

x = floor(linspace(1,N,20));
zeta_knots = zeros(L,k,length(x));

for ii=1:length(x)
    inds_ii = [max(1,x(ii)-N/10):min(N,x(ii)+N/10)];
    cov_est_ii = cov(y(:,inds_ii)') + 0*ones(p); % - Sig_mat;
    C = chol(cov_est_ii)';
    C_ii = C(:,1:k);
    zeta_knots(:,:,ii) = theta \ C_ii;
    Sig_est(:,ii) = diag(C_ii*C_ii');
    
end

zeta_spline = spline(x,zeta_knots);
zeta = ppval([1:N],zeta_spline);

return;


function xi = sample_xi_init(y,invSig_vec,psi,temp)

[p N] = size(y);
[k N] = size(psi);

invSigMat = invSig_vec(ones(k,1),:);

xi = zeros(k,N);

for nn=1:N
    inds_nn = [max(1,nn-N/10):min(N,nn+N/10)];
    cov_est_nn = cov(y(:,inds_nn)') + 0*ones(p); % - Sig_mat;
    C = chol(cov_est_nn)';
    C_nn = C(:,1:k);
    theta_zeta_n = C_nn;
    y_tilde_n = y(:,nn)-theta_zeta_n*psi(:,nn);
    zeta_theta_invSig = theta_zeta_n'.*invSigMat;  % zeta_theta_invSig = zeta(:,:,nn)'*(theta'*invSig);
    
    cholSig_xi_n_trans = chol(eye(k) + zeta_theta_invSig*theta_zeta_n) \ eye(k);  % Sig_eta_n = inv(eye(k) + zeta_theta_invSig*theta_zeta_nn);
    m_xi_n = cholSig_xi_n_trans*(cholSig_xi_n_trans'*(zeta_theta_invSig*y_tilde_n));  % m_eta_n = Sig_eta_n*(zeta_theta_invSig*y(:,nn));
    
    xi(:,nn) = m_xi_n + temp*cholSig_xi_n_trans*randn(k,1); % eta(:,nn) = m_eta_n + chol(Sig_eta_n)'*randn(k,1);
end

return;


function logp = normpdfln(x,mu,Sigma,iSigma)

if nargin<4
    q = size(Sigma,1);
    iSigma = Sigma \ eye(q);
    
    logdet_iSigma = -2*sum(log(diag(chol(Sigma))));
else
    logdet_iSigma = 2*sum(log(diag(chol(iSigma))));
end

logp = 0.5*logdet_iSigma - 0.5*(x-mu)'*(iSigma*(x-mu));

return;