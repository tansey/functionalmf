% BNP_covreg_varinds(y,prior_params,settings,restart,true_params)
%
% Method for producing samples of predictor-dependent mean and covariance
% in the presence of missing data.  If no data are missing, use BNP_covreg.
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

function BNP_covreg_varinds(y,prior_params,settings,restart,true_params)

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

% K represents the correlation matrix of the latent GPs.  There are three
% options for how to handle K during sampling, as determined below:
% (NOTE: CURRENTLY, THIS VERSION ONLY SUPPORTS FIXING K.  SEE BNP_covreg.m
% FOR CODE WHERE SAMPLING K IS POSSIBLE.)
sample_K_flag = settings.sample_K_flag; % (1) sample K marginalizing zeta, (2) sample K conditioning on zeta, (3) set K to fixed value

% Indicate whether or not we wish to model a latent mean \mu(x):
latent_mean = settings.latent_mean;

% Indices of observations present:
inds_y = settings.inds_y;

y(~inds_y) = 0;

k = settings.k;  % dimension of latent factors
L = settings.L;  % number of dictionary elements = L*k:
Niter = settings.Niter;   % number of Gibbs iterations
trial = settings.trial;   % label for MCMC chain

if ~restart
    
    % Initialize structure for storing samples:
    Stats(1:settings.saveEvery/settings.storeEvery) = struct('zeta',zeros(L,k,N),'psi',zeros(k,N),'invSig_vec',zeros(1,p),...
        'theta',zeros(p,L),'eta',zeros(k,N),'phi',zeros(p,L),'tau',zeros(1,L),'K_ind',0);
    store_counter = 1;
    
    % Sample hyperparams from prior:
    delta = zeros(1,L);
    delta(1) = gamrnd(prior_params.hypers.a1,1);
    delta(2:L) = gamrnd(prior_params.hypers.a2*ones(1,L-1),1);
    tau = exp(cumsum(log(delta)));
    phi = gamrnd(prior_params.hypers.a_phi*ones(p,L),1) / prior_params.hypers.b_phi;
    
    % Sample theta, eta, and Sigma initially as prior draws:
    theta = zeros(p,L);
    for pp=1:p
        theta(pp,:) = chol(diag(1./(phi(pp,:).*tau)))'*randn(L,1);
    end

    xi = randn(k,N);
    psi = zeros(k,N);
    eta = psi + xi;
    
    % invSig_vec represents the diagonal elements of \Sigma_0^{-1}:
    invSig_vec = gamrnd(prior_params.sig.a_sig*ones(1,p),1) / prior_params.sig.b_sig;
    
    % Sample initial GP cov K and GP latent functions zeta_i:
    if sample_K_flag==1 || sample_K_flag==2
        Pk = cumsum(prior_params.K.c_prior);
        K_ind = 1 + sum(Pk(end)*rand(1) > Pk);
    else
        K_ind = 1;
    end
    % Sample zeta_i using initialization scheme based on data and other
    % sampled params:
    zeta = zeros(L,k,N);
    zeta = sample_zeta(y,theta,eta,invSig_vec,zeta,prior_params.K.invK(:,:,K_ind),inds_y);
    
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
    if nargin>4
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
    invSig_vec = sample_sig(y,theta,eta,zeta,prior_params.sig,inds_y);
    
    % Sample weightings matrix hyperparameters given weightings matrix:
    [phi tau] = sample_hypers(theta,phi,tau,prior_params.hypers);
    
    % Sample weightings matrix given the data, latent factors, latent GP
    % functions, noise parameters, and hyperparameters:
    theta = sample_theta(y,eta,invSig_vec,zeta,phi,tau,inds_y);
    
    % Sample K based on the p x N dimensional Gaussian posterior
    % p(K | y, theta, eta, Sigma_0) having marginalized the latent GP
    % functions zeta.  If p x N is too large, sample the GP cov matrix K
    % given the latent GP functions zeta p(K | zeta) (yielding K cond ind
    % of everything else, though leading to much slower mixing rates):
    if sample_K_flag == 1
        error('need to code up K sampling for inds_y')
        %K_ind = sample_K_marg_zeta(y,theta,eta,invSig_vec,prior_params.K,K_ind);
    elseif sample_K_flag == 2
        error('need to code up K sampling for inds_y')
        %K_ind = sample_K_cond_zeta(zeta,prior_params.K);
    else
        K_ind = 1;  % set K to fixed value (i.e., do not resample hyperparameters)
    end
    
    % The following represents a block sampling of \psi and \xi.
    % If modeling a non-zero latent mean \mu(x), sample the latent mean
    % GPs \psi(x) marginalizing \xi.  Otherwise, set these elements to zeros:
    if latent_mean
        psi = sample_psi_margxi(y,theta,invSig_vec,zeta,psi,prior_params.K.invK(:,:,K_ind),inds_y);
    else
        psi = zeros(size(xi));
    end
    % Sample latent factors given the data, weightings matrix, latent GP
    % functions, and noise parameters.
    xi = sample_xi(y,theta,invSig_vec,zeta,psi,inds_y);
        
    eta = psi + xi;
    
    % Sample latent GP functions zeta_i given the data, weightings matrix,
    % latent factors, noise params, and GP cov matrix (hyperparameter):
    
    for ii=1:num_iters % one can cycle through this sampling stage multiple times by adjusting num_iters
        zeta = sample_zeta(y,theta,eta,invSig_vec,zeta,prior_params.K.invK(:,:,K_ind),inds_y);
    end
    
    num_iters = 1;    
    
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
        
        store_counter = store_counter + 1;
        disp(nn);
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

function zeta = sample_zeta(y,theta,eta,invSig_vec,zeta,invK,inds_y)

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
error_nn = zeros(L,N);
for nn=1:N
    mu_tot(:,nn) = theta*zeta(:,:,nn)*eta(:,nn);
    % Store the amount that will be added, but shouldn't be because of
    % missing observations:
    error_nn(:,nn) = (theta.^2)'*(invSig_vec'.*(1-inds_y(:,nn)));
end 

%if sum(zeta(:))==0
%    numiter = 50;
%else
    numiter = 1;
%end
    
for nn=1:numiter
    for ll=1:L  % walk through each row of zeta matrix sequentially
        theta_ll = theta(:,ll);
        for kk=randperm(k);  % create random ordering for kk in sampling zeta_{ll,kk}
            eta_kk = eta(kk,:);
            zeta_ll_kk = squeeze(zeta(ll,kk,:))';
            mu_tot = mu_tot - theta_ll(:,ones(1,N)).*eta_kk(ones(p,1),:).*zeta_ll_kk(ones(p,1),:);
            
            % Using standard Gaussian identities, form posterior of
            % zeta_{ll,kk} using information form of Gaussian prior and likelihood:
            A_lk_invSig_A_lk = (eta(kk,:).^2).*((theta(:,ll).^2)'*invSig_vec'-error_nn(ll,:));
            
            theta_tmp = theta(:,ll)'.*invSig_vec;
            ytilde = (y - mu_tot).*inds_y;  % normalize data by subtracting mean of tilde(eps)
            theta_lk = eta(kk,:)'.*(theta_tmp*ytilde)'; % theta_lk = eta(kk,:)'.*diag(theta_tmp(ones(1,N),:)*ytilde);
            
            % Transform information parameters:
            cholSig_lk_trans = chol(invK + diag(A_lk_invSig_A_lk)) \ eye(N);  % Sig_lk = inv(invK + A_lk_invSig_A_lk);
            m_lk = cholSig_lk_trans*(cholSig_lk_trans'*theta_lk);  % m_lk = Sig_lk*theta_lk;
            
            % Sample zeta_{ll,kk} from posterior Gaussian:
            zeta(ll,kk,:) = m_lk + cholSig_lk_trans*randn(N,1);  % zeta(ll,kk,:) = m_lk + chol(Sig_lk)'*randn(N,1);
            
            zeta_ll_kk = squeeze(zeta(ll,kk,:))';
            mu_tot = mu_tot + theta_ll(:,ones(1,N)).*eta_kk(ones(p,1),:).*zeta_ll_kk(ones(p,1),:);
        end
    end
end

return;


function psi = sample_psi_margxi(y,theta,invSig_vec,zeta,psi,invK,inds_y)

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
OmegaInvOmegaOmegaSigma0 = zeros(k,p,N);
for nn=1:N
    Omega(inds_y(:,nn),:,nn) = theta(inds_y(:,nn),:)*zeta(:,:,nn);
    temp = (Omega(inds_y(:,nn),:,nn)*Omega(inds_y(:,nn),:,nn)'...
        + Sigma_0(inds_y(:,nn),inds_y(:,nn))) \ eye(sum(inds_y(:,nn)));
    OmegaInvOmegaOmegaSigma0(:,inds_y(:,nn),nn) = Omega(inds_y(:,nn),:,nn)'*temp;
    mu_tot(:,nn) = Omega(:,:,nn)*psi(:,nn);  % terms will be 0 where inds_y(:,nn)=0
end

if sum(sum(mu_tot))==0 % if this is a call to initialize psi
    numTotIters = 50;
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
        psi(kk,:) = m_k + cholSig_k_trans*randn(N,1);  % zeta(ll,kk,:) = m_lk + chol(Sig_lk)'*randn(N,1);
        
        psi_kk = psi(kk,:);
        mu_tot = mu_tot + Omega_kk.*psi_kk(ones(p,1),:);

    end
    
end

return;

function xi = sample_xi(y,theta,invSig_vec,zeta,psi,inds_y)

% Sample latent factors eta_i using standard Gaussian identities based on
% the fact that:
%
% y_i = (theta*zeta)*eta_i + eps_i,   eps_i \sim N(0,Sigma_0),   eta_i \sim N(0,I) 
%
% and using the information form of the Gaussian likelihood and prior.

[p N] = size(y);
[L k] = size(zeta(:,:,1));

xi = zeros(k,N);
for nn=1:N
    theta_zeta_n = theta*zeta(:,:,nn);
    y_tilde_n = y(:,nn)-theta_zeta_n*psi(:,nn);
    
    invSigMat = invSig_vec.*inds_y(:,nn)';
    invSigMat = invSigMat(ones(k,1),:);
    zeta_theta_invSig = theta_zeta_n'.*invSigMat;  % zeta_theta_invSig = zeta(:,:,nn)'*(theta'*invSig);

    cholSig_xi_n_trans = chol(eye(k) + zeta_theta_invSig*theta_zeta_n) \ eye(k);  % Sig_eta_n = inv(eye(k) + zeta_theta_invSig*theta_zeta_nn);   
    m_xi_n = cholSig_xi_n_trans*(cholSig_xi_n_trans'*(zeta_theta_invSig*y_tilde_n));  % m_eta_n = Sig_eta_n*(zeta_theta_invSig*y(:,nn));
    
    xi(:,nn) = m_xi_n + cholSig_xi_n_trans*randn(k,1); % eta(:,nn) = m_eta_n + chol(Sig_eta_n)'*randn(k,1);
end

return;

function theta = sample_theta(y,eta,invSig_vec,zeta,phi,tau,inds_y)

[p N] = size(y);
L = size(zeta,1);
theta = zeros(p,L);

eta_tilde = zeros(L,N);
for nn=1:N
    eta_tilde(:,nn) = zeta(:,:,nn)*eta(:,nn);
end
eta_tilde = eta_tilde';

for pp=1:p
    inds_y_p = inds_y(pp,:)';
    eta_tilde_p = eta_tilde.*inds_y_p(:,ones(1,L));
    chol_Sig_theta_p_trans = chol(diag(phi(pp,:).*tau) + invSig_vec(pp)*(eta_tilde_p'*eta_tilde_p)) \ eye(L);
    m_theta_p = invSig_vec(pp)*(chol_Sig_theta_p_trans*chol_Sig_theta_p_trans')*(eta_tilde_p'*y(pp,:)');
    theta(pp,:) = m_theta_p + chol_Sig_theta_p_trans*randn(L,1);
end

return;


function invSig_vec = sample_sig(y,theta,eta,zeta,prior_params,inds_y)

[p N] = size(y);

a_sig = prior_params.a_sig;
b_sig = prior_params.b_sig;

inds_vec = [1:N];

invSig_vec = zeros(1,p);
for pp = 1:p
    sq_err = 0;
    for nn=inds_vec(inds_y(pp,:))
        sq_err = sq_err + (y(pp,nn) - theta(pp,:)*zeta(:,:,nn)*eta(:,nn))^2;
    end
    invSig_vec(pp) = gamrnd(a_sig + 0.5*sum(inds_y(pp,:)),1) / (b_sig + 0.5*sq_err);
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
