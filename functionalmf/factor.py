import abc
import numpy as np
import scipy as sp
import warnings
from scipy.sparse import kron, eye, spdiags, csc_matrix, block_diag, diags, coo_matrix
from scipy.stats import norm, multivariate_normal as mvn, gamma
from scipy.special import gammaln
from scipy.linalg import solve_triangular
from sksparse.cholmod import cholesky
from functionalmf.genlasso import _BayesianModel, \
                                 ConjugateInverseGammaPrior
from functionalmf.utils import bayes_delta, bayes_grid_penalty,\
                              moving_average,\
                              sample_horseshoe_plus, sample_horseshoe, ilogit
from functionalmf.fast_mvn import sample_mvn_from_precision
from functionalmf.elliptical_slice import elliptical_slice, elliptical_slice_
from functionalmf.gass import gass
from concurrent import futures
from multiprocessing import Pool
import SharedArray as sa


class BayesianTensorFiltering(_BayesianModel):
    def __init__(self, nrows, ncols, ndepth,
                       nembeds=5, tf_order=2,
                       sigma2_init=None, sigma2_true=None,
                       sigma2_a=0.1, sigma2_b=0.1,
                       lam2_init=None, lam2_true=None,
                       Tau2_init=None, Tau2_true=None,
                       W_init=None, V_init=None,
                       W_true=None, V_true=None,
                       stability=1e-6, 
                       force_psd=True, 
                       force_psd_eps=1e-6,
                       force_psd_attempts=4,
                       **kwargs):
        super().__init__(**kwargs)
        self.nrows = nrows
        self.ncols = ncols
        self.ndepth = ndepth
        self.nembeds = nembeds
        self.stability = stability
        self.linalg_opts = dict(
            force_psd=force_psd,
            force_psd_eps=force_psd_eps,
            force_psd_attempts=force_psd_attempts
        )

        # Setup the trend filtering prior
        self.Delta = bayes_grid_penalty(ndepth, tf_order)

        # Row embedding variance
        self.sigma2_a = sigma2_a
        self.sigma2_b = sigma2_b
        self.sigma2_model = ConjugateInverseGammaPrior(1, self.sigma2_a, self.sigma2_b)
        if sigma2_true is not None:
            self.sigma2 = sigma2_true
            self.sample_sigma2 = False
        else:
            self.sample_sigma2 = True
            if sigma2_init is not None:
                self.sigma2 = sigma2_init
            else:
                self._init_sigma2()

        # Global shrinkage prior
        if lam2_true is not None:
            self.lam2 = lam2_true
            self.sample_lam2 = False
        else:
            self.sample_lam2 = True
            self._init_lam2()
            if lam2_init is not None:
                self.lam2 = lam2_init

        # Local group shrinkage priors
        if Tau2_true is not None:
            self.Tau2 = Tau2_true.copy()
            self.sample_Tau2 = False
        else:
            self.sample_Tau2 = True
            if Tau2_init is not None:
                self.Tau2 = Tau2_init.copy()
            else:
                self._init_Tau2()
        assert self.Tau2.shape == (self.ncols, self.Delta.shape[0])

        # Row embeddings
        if W_true is not None:
            self.W = W_true.copy()
            self.sample_W = False
        else:
            self.sample_W = True
            if W_init is not None:
                self.W = np.copy(W_init)
            else:
                self._init_W()
        assert self.W.shape == (nrows, nembeds)

        # Column functional embeddings
        if V_true is not None:
            self.V = V_true.copy()
            self.sample_V = False
        else:
            self.sample_V = True
            if V_init is not None:
                self.V = V_init.copy()
            else:
                self._init_V()
        assert self.V.shape == (ncols, ndepth, nembeds)

    def resample(self, data, **kwargs):
        # Sample the row-wise embedding variance
        if self.sample_sigma2:
            self._resample_sigma2()

        # Update the local shrinkage variables
        if self.sample_Tau2:
            self._resample_Tau2()

        if self.sample_lam2:
            self._resample_lam2()
            
        if self.sample_W:
            self._resample_W(data)

        if self.sample_V:
            self._resample_V(data)

    def _resample_sigma2(self):
        W_vec, _ = self._pack_W(self.W)
        self.sigma2 = 1/self.sigma2_model.resample((np.zeros_like(W_vec), W_vec))

    def _resample_Tau2(self):
        for j in range(self.ncols):
            deltas = self.Delta.dot(self.V[j])
            rate = (deltas**2).sum(axis=1) / (2*self.lam2) + 1/self.Tau2_c[j].clip(self.stability, 1/self.stability)
            self.Tau2[j] = 1/np.random.gamma((self.nembeds + 1) / 2, 1/rate.clip(self.stability, 1/self.stability))
            self.Tau2_c[j] = 1/np.random.gamma(1, 1 / (1/self.Tau2[j] + 1/self.Tau2_b[j]).clip(self.stability, 1/self.stability))
            self.Tau2_b[j] = 1/np.random.gamma(1, 1 / (1/self.Tau2_c[j] + 1/self.Tau2_a[j]).clip(self.stability, 1/self.stability))
            self.Tau2_a[j] = 1/np.random.gamma(1, 1 / (1/self.Tau2_b[j] + 1).clip(self.stability, 1/self.stability))

    def _resample_lam2(self):
        # TODO: make lam2 be tf_k specific. so for an order 2 model, you have
        # a separate lam2 for the anchor (fixed to be very large), one for
        # the TV penalty, one for the linear penalty, and one for the quadratic
        rate = 1/self.lam2_a
        for j in range(self.ncols):
            deltas = self.Delta.dot(self.V[j])
            rate = ((deltas / np.sqrt(self.Tau2[j])[:,None])**2).sum() / 2
        shape = (self.Delta.shape[0] * self.ncols * self.nembeds + 1)
        self.lam2 = max(1e-5, 1/np.random.gamma(shape/2, 1/rate)) # Clip lam2 at some reasonable value
        self.lam2_a = 1/np.random.gamma(1, 1 / (1/self.lam2 + 1))

    def _pack_W(self, W):
        # W is a lower-triangular matrix with isotropic variance
        if self.nrows >= self.nembeds:
            w_tril_size = (self.nembeds**2 - self.nembeds) // 2 + self.nembeds
            w_dense_size = (self.nrows - self.nembeds)*self.nembeds
        else:
            w_tril_size = (self.nrows**2 - self.nrows) // 2 + self.nrows
            w_dense_size = 0
        w_len = w_tril_size + w_dense_size

        # The full set of current values for all embeddings
        cur = np.full(w_len, np.nan)

        # Get the W embeddings
        w_vec = np.full(w_len, 1/self.sigma2)
        W_precision = spdiags(w_vec, 0, w_len, w_len, format='csc')
        cur[:w_tril_size] = W[np.tril_indices(min(self.nembeds, self.nrows))]
        cur[w_tril_size:w_len] = W[min(self.nembeds, self.nrows):].flatten()

        return cur, W_precision

    def _pack_V(self, V):
        blocks = []

        # The full set of current values for all embeddings
        cur = np.full(np.prod(self.V.shape), np.nan)

        # V is a block-diagonal covariance matrix for each embedding V_j
        Dt = self.Delta.T.tocsc()
        I_embed = eye(self.nembeds, format='csc')
        for j in range(self.ncols):
            lam_Tau = spdiags(1/ (self.lam2 * self.Tau2[j]), 0, self.Tau2.shape[1], self.Tau2.shape[1], format='csc')
            Vj_prior = kron(I_embed, self.Delta.T.tocsc().dot(lam_Tau).dot(self.Delta)).tocsc()
            blocks.append(Vj_prior)
            cur[j*self.ndepth*self.nembeds:(j+1)*self.ndepth*self.nembeds] = V[j].T.flatten()

        # Form a joint block-diagonal precision matrix for all embeddings thus far
        Q = block_diag(blocks, format='csc')

        return cur, Q

    def _pack_embeddings(self, W, V):
        W_vec, W_precision = self._pack_W(W)
        V_vec, V_precision = self._pack_V(V)
        return np.concatenate([W_vec, V_vec]), block_diag([W_precision, V_precision], format='csc')

    def _unpack_W(self, vec, W):
        # W is a lower-triangular matrix with isotropic variance
        if self.nrows >= self.nembeds:
            w_tril_size = (self.nembeds**2 - self.nembeds) // 2 + self.nembeds
            w_dense_size = (self.nrows - self.nembeds)*self.nembeds
        else:
            w_tril_size = (self.nrows**2 - self.nrows) // 2 + self.nrows
            w_dense_size = 0
        w_len = w_tril_size + w_dense_size

        # Unpack W as a lower-triangular matrix
        W[np.tril_indices(min(self.nembeds, self.nrows))] = vec[:w_tril_size]
        W[min(self.nembeds, self.nrows):] = vec[w_tril_size:w_len].reshape((self.nrows-self.nembeds, self.nembeds))

    def _unpack_V(self, vec, V):
        # Unpack V
        for j in range(self.ncols):
            vec_j = vec[j*self.ndepth*self.nembeds:(j+1)*self.ndepth*self.nembeds]
            V[j] = vec_j.reshape((self.nembeds, self.ndepth)).T

    def _unpack_embeddings(self, vec, W, V):
        # W is a lower-triangular matrix with isotropic variance
        w_tril_size = (self.nembeds**2 - self.nembeds) // 2 + self.nembeds
        w_full_size = (self.nrows - self.nembeds)*self.nembeds
        w_len = w_tril_size + w_full_size

        self._unpack_W(vec[:w_len], W)
        self._unpack_V(vec[w_len:], V)

    def _init_W(self):
        self.W = np.random.normal(0, np.sqrt(self.sigma2), size=(self.nrows, self.nembeds))
        if self.nrows > 1:
            self.W[np.triu_indices(self.nembeds, k=1)] = 0

    def _init_V(self):
        self.V = np.full((self.ncols, self.ndepth, self.nembeds), np.nan)
        I_embed = eye(self.nembeds, format='csc')
        for j in range(self.ncols):
            lam_Tau = spdiags(1/ (self.lam2 * self.Tau2[j]), 0, self.Tau2.shape[1], self.Tau2.shape[1], format='csc')
            Q = kron(I_embed, self.Delta.T.tocsc().dot(lam_Tau).dot(self.Delta)).tocsc()
            self.V[j] = sample_mvn_from_precision(Q, **self.linalg_opts).reshape((self.nembeds, self.ndepth)).T
        self.V = self.V.clip(-10,10)

    def _init_Tau2(self):
        self.Tau2, self.Tau2_c, self.Tau2_b, self.Tau2_a = sample_horseshoe_plus(size=(self.ncols, self.Delta.shape[0]))
        self.Tau2 = self.Tau2.clip(0,9)

    def _init_lam2(self):
        self.lam2, self.lam2_a = sample_horseshoe()
        self.lam2 = self.lam2.clip(0,4)

    def _init_sigma2(self):
        self.sigma2 = 1/self.sigma2_model.draw_from_prior()

    def _inferred_variables(self, var_map):
        var_map['W'] = np.copy(self.W)
        var_map['V'] = np.copy(self.V)
        var_map['sigma2'] = self.sigma2
        var_map['lam2'] = self.lam2
        var_map['Tau2'] = np.copy(self.Tau2)

    def logprob(self, Y, **kwargs):
        Mu = np.matmul(model.W[None], np.transpose(model.V, [0,2,1])).transpose([1,0,2])
        return norm.logpdf(Y, Mu, scale=np.sqrt(self.sigma2))


    def _default_hyperparam_options(self, hyperparams, lam2=None,
                                    min_lam2=1e-6, max_lam2=1e3, num_lam2=10,
                                    **kwargs):
        if lam is None:
            hyperparams['lam2'] = np.exp(np.linspace(np.log(min_lam2),
                                                          np.log(max_lam2),
                                                          num_lam2))[::-1]
        else:
            hyperparams['lam2'] = lam2

    def _set_hyperparameters(self, hyperparams):
        self.lam2 = hyperparams['lam2']

    def _resample_W(self, data):
        raise NotImplementedError

    def _resample_V(self, data):
        raise NotImplementedError

class GaussianBayesianTensorFiltering(BayesianTensorFiltering):
    def __init__(self, nrows, ncols, ndepth,
                       nu2_init=None, nu2_true=None,
                       nu2_a=0.1, nu2_b=0.1, **kwargs):
        super().__init__(nrows, ncols, ndepth, **kwargs)

        # Observation noise (variance)
        self.nu2_a = nu2_a
        self.nu2_b = nu2_b
        self.nu2_model = ConjugateInverseGammaPrior(1, self.nu2_a, self.nu2_b)
        if nu2_true is not None:
            self.nu2 = nu2_true
            self.sample_nu2 = False
        else:
            self.sample_nu2 = True
            if nu2_init is not None:
                self.nu2 = nu2_init
            else:
                self._init_nu2()

    def resample(self, data):
        # Sample the observation noise
        if self.sample_nu2:
            self._resample_nu2(data)

        super().resample(data)

    def _resample_W(self, data):
        Y = data

        assert len(Y.shape) == 3 or len(Y.shape) == 4, 'Observations must be 3- or 4-tensor.'

        # If there is missing data, or we are dealing with different 
        # dimensions of embeddings, recompute the posterior parameters
        recompute = np.any(np.isnan(Y))

        # If there are no replicates, make this a one-replicate case
        if len(Y.shape) == 3:
            Y = Y[...,None]
        
        # Get the sufficient statistics for the observations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            Y_counts = (~np.isnan(Y)).sum(axis=-1)
            Y_mean = np.nanmean(Y,axis=-1)
        
        # Update W one embedding at a time
        for i in range(self.nrows):
            # Vectorize Y
            Y_flat = Y_mean[i].flatten()
            Y_counts_flat = Y_counts[i].flatten()

            # Remove missing entries
            missing = np.isnan(Y_flat)
            Y_flat = Y_flat[~missing]

            # Handle both homoskedastic and heteroskedastic errors
            if np.isscalar(self.nu2):
                Y_counts_flat = Y_counts_flat[~missing] / self.nu2
            else:
                Y_counts_flat = Y_counts_flat[~missing] / self.nu2[i].flatten()[~missing]

            # W is lower triangular, so don't update the upper triangular entries of W
            if i < self.nembeds or recompute:
                # Vectorize V and remove entries where we have missing data
                V_flat = self.V.reshape((-1, self.nembeds))[:,0:min(i+1, self.nembeds)]
                V_flat = V_flat[~missing]

                # Calculate the mean and precision of the posterior
                Xt = (V_flat * Y_counts_flat[:,None]).T
                Q = Xt.dot(V_flat) + eye(min(i+1,self.nembeds), format='csc')/self.sigma2
                Lt = np.linalg.cholesky(Q).T

            # Sample from the posterior
            mu = Xt.dot(Y_flat)
            z = np.random.normal(size=min(i+1,self.nembeds))
            self.W[i,:min(i+1,self.nembeds)] = sp.linalg.cho_solve((Lt, False), mu) + solve_triangular(Lt, z, lower=False)

    def _resample_V(self, data):
        Y = data

        # If there are no replicates, make this a one-replicate case
        if len(Y.shape) == 3:
            Y = Y[...,None]

        # Get the sufficient statistics for the observations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            Y_counts = (~np.isnan(Y)).sum(axis=-1)
            Y_mean = np.nanmean(Y,axis=-1)

        # Update V one column of embeddings at a time
        for j in range(self.ncols):
            # Vectorize Y
            Y_flat = Y_mean[:,j].flatten()
            Y_counts_flat = Y_counts[:,j].flatten()

            # Remove missing entries
            missing = np.isnan(Y_flat)
            Y_flat = Y_flat[~missing]
            
            # Handle both homoskedastic and heteroskedastic errors
            if np.isscalar(self.nu2):
                Y_counts_flat = diags(Y_counts_flat[~missing] / self.nu2 , 0)
            else:
                Y_counts_flat = diags(Y_counts_flat[~missing] / self.nu2[:,j].flatten()[~missing], 0)

            # Recalculate if there is a new missing data pattern
            if j == 0 or np.any(sparsity_pattern != missing):
                # Treat the W embeddings as covariates in a linear regression
                X = kron(self.W, eye(self.ndepth, format='csc'), format='csc')[~missing]
                Xt = X.T.dot(Y_counts_flat).tocsc()
                Q_likelihood = Xt.dot(X)
                I_embed = eye(self.nembeds, format='csc')
                sparsity_pattern = missing
            mu = Xt.dot(Y_flat)

            # Get the global-local shrinkage prior as a diagonal precision matrix
            lam_Tau = spdiags(1/ (self.lam2 * self.Tau2[j]), 0, self.Tau2.shape[1], self.Tau2.shape[1], format='csc')
            Q_prior = kron(I_embed, self.Delta.T.tocsc().dot(lam_Tau).dot(self.Delta)).tocsc()

            # Ridge regression 
            Q = Q_likelihood + Q_prior
            self.V[j] = sample_mvn_from_precision(Q, mu_part=mu, sparse=True, **self.linalg_opts).reshape((self.nembeds, self.ndepth)).T

    def _resample_nu2(self, data):
        Y = data
        Mu = np.einsum('nk,mtk->nmt', self.W, self.V)
        while len(Y.shape) > len(Mu.shape):
            Mu = Mu[...,None]
        self.nu2 = 1/self.nu2_model.resample((Mu, Y))

    def _init_nu2(self):
        self.nu2 = 1/self.nu2_model.draw_from_prior()

    def _inferred_variables(self, var_map):
        super()._inferred_variables(var_map)
        var_map['nu2'] = self.nu2

class BinomialBayesianTensorFiltering(GaussianBayesianTensorFiltering):
    def __init__(self, nrows, ncols, ndepth,
                       pg_seed=42, **kwargs):
        super().__init__(nrows, ncols, ndepth, **kwargs)

        # Initialize the Polya-Gamma sampler
        from pypolyagamma import PyPolyaGamma
        self.pg = PyPolyaGamma(seed=pg_seed)
        self.nu2 = np.zeros((nrows, ncols, ndepth))
        self.nu2_flat = np.zeros(np.prod(self.nu2.shape))
        self.sample_nu2 = True

    def _resample_W(self, data):
        Y, N = data
        kappa = (Y - N/2) * self.nu2
        super()._resample_W(kappa)

    def _resample_V(self, data):
        Y, N = data
        kappa = (Y - N/2) * self.nu2
        super()._resample_V(kappa)

    def _resample_nu2(self, data):
        '''Update the latent variables, which lead to variance terms in the
        gaussian sampler steps.'''
        Y, N = data
        Mu = np.einsum('nk,mtk->nmt', self.W, self.V)
        # missing = np.isnan(Y)
        # for s in np.ndindex(Y.shape):
        #     if missing[s]:
        #         continue
        #     self.nu2[s] = 1/self.pg.pgdraw(N[s], Mu[s])
        # print(N.flatten()[:5], Mu.flatten()[:5], self.nu2_flat[:5])
        with np.errstate(divide='ignore'):
            self.pg.pgdrawv(N.flatten(), Mu.flatten(), self.nu2.reshape(-1))
            self.nu2 = 1/self.nu2


class NegativeBinomialBayesianTensorFiltering(BinomialBayesianTensorFiltering):
    def __init__(self, nrows, ncols, ndepth,
                    R_true=None, # Treat R as an unknown random variable that must be sampled
                    R_init=None, # Initial value of R, if R_true is None
                    nmetropolis=30, # Random-walk metropolis hastings steps per Gibbs step for rate parameter
                    rpropstdev=0.1, # proposal standard deviation in RW-MH algorithm
                    rstdev=1, # Prior standard deviation on log(R), defaults to 1 (in log space)
                    rdims=(0,1,2), # Dims for which to aggregate rate parameters along; default is all 3
                        **kwargs):
        super().__init__(nrows, ncols, ndepth, **kwargs)
        # self.R = np.random.([(1 if i in rdims else c) for i,c in enumerate([nrows, ncols, ndepth])])
        # self.N = np.ones((nrows, ncols, ndepth))

        # MCMC-within-Gibbs parameters for updating R
        if R_true is not None:
            self.sample_R = False
            self.R = R_true
        else:
            self.sample_R = True
            self.nmetropolis = nmetropolis
            self.rpropstdev = rpropstdev
            self.rstdev = rstdev
            # Dims to aggregate over
            self.rdims = [3] + (sorted([d for d in rdims])[::-1] if rdims is not None else [])
            if R_init is None:
                self._init_R()
            else:
                self.R = R_init

    

    def resample(self, data):
        # If no replicate, treat this as 1 replicate
        if len(data.shape) == 3:
            data = data[...,None]

        # Track the missing values so we don't put false zeros in
        missing = np.all(np.isnan(data), axis=-1)

        # Sample the NB rate parameter
        if self.sample_R:
            self._resample_R(data)

        # Binomial update step-- can sum the independent trials and successes
        Y = np.nansum(data, axis=-1)
        Y[missing] = np.nan

        # Run the Binomial sampler steps
        super().resample((Y,self.N))

    def _resample_R(self, data):
        '''Random-walk MCMC for R'''
        self.R = self.R[...,None]
        logR = np.log(self.R)

        # Current success probability
        P = ilogit(np.einsum('nk,mtk->nmt', self.W, self.V).clip(-10,10))[...,None]

        # Run a fixed number of random walk Metropolis-Hastings steps
        for mcmc_step in range(self.nmetropolis):
            # Sample a candidate R values
            candidate_logR = logR + np.random.normal(0, self.rpropstdev, size=logR.shape)
            candidate_R = np.exp(candidate_logR)

            # Get the prior log-likelihood ratio
            accept_prior = norm.logpdf(candidate_logR, loc=0, scale=self.rstdev) - norm.logpdf(logR, loc=0, scale=self.rstdev)

            # Get the likelihood log-likelihood ratio
            accept_likelihood = (gammaln(data+candidate_R) - gammaln(candidate_R) - gammaln(data+self.R) + gammaln(self.R) # combinatorial bit
                                 + (candidate_R-self.R)*np.log(1-P)) # failure prob bit; success prob bit does not use R
            
            # Multiply replicates and any other aggregated likelihoods
            for dim in self.rdims:
                accept_likelihood = np.nansum(accept_likelihood, axis=dim)

            # Get rid of aggregated prior dims; make sure to go back to 1 for any size-1 unaggregated dims
            accept_prior = np.squeeze(accept_prior).reshape(accept_likelihood.shape)

            # Get the acceptance probability for each R
            accept_probs = np.exp(np.clip(accept_prior + accept_likelihood, -10, 1)).reshape(self.R.shape)

            # Accept/reject the samples
            accept_indices = np.random.random(size=accept_probs.shape) <= accept_probs

            accept_indices = accept_indices & (candidate_R > 1) # TEMP
            accepted_logR = candidate_logR[accept_indices]
            logR[accept_indices] = accepted_logR
            self.R[accept_indices] = np.exp(accepted_logR)

        # Update the pseudo-counts for the PG-augmentation step
        self.N = np.nansum(data + self.R, axis=-1) # Binomial N can sum independent trials
        self.R = self.R[...,0]

    def _inferred_variables(self, var_map):
        super()._inferred_variables(var_map)
        var_map['R'] = self.R

    def _init_R(self):
        R_size = [(1 if i in self.rdims else c) for i,c in enumerate([self.nrows, self.ncols, self.ndepth])]
        self.R = np.exp(np.random.normal(0, self.rstdev, size=R_size))
        self.R += 1 # TEMP
        


class NonconjugateBayesianTensorFiltering(BayesianTensorFiltering):
    def __init__(self, nrows, ncols, ndepth, loglikelihood, **kwargs):
        super().__init__(nrows, ncols, ndepth, **kwargs)
        self.loglikelihood = loglikelihood

    def _resample_W(self, data):
        cur, Q = self._pack_W(self.W)

        # Draw from the prior
        prior_sample = sample_mvn_from_precision(Q, sparse=True, **self.linalg_opts)
            
        # Draw from the posterior
        sample, _ = elliptical_slice_(cur, prior_sample,
                                       self._ess_W_loglikelihood,
                                       ll_args=data)
        self._unpack_W(sample, self.W)

    def _resample_V(self, data):
        cur, Q = self._pack_V(self.V)
        prior_sample = sample_mvn_from_precision(Q, sparse=True, **self.linalg_opts)
        sample, _ = elliptical_slice_(cur, prior_sample,
                                       self._ess_V_loglikelihood,
                                       ll_args=data)
        self._unpack_V(sample, self.V)

    def _ess_W_loglikelihood(self, sample, data):
        W = np.zeros_like(self.W)
        self._unpack_W(sample, W)
        return self.loglikelihood(W, self.V, data)

    def _ess_V_loglikelihood(self, sample, data):
        V = np.zeros_like(self.V)
        self._unpack_V(sample, V)
        return self.loglikelihood(self.W, V, data)

    def _ess_joint_loglikelihood(self, sample, data):
        '''Unpack the embeddings and pass to the black box likelihood'''
        W = np.zeros_like(self.W)
        V = np.zeros_like(self.V)
        self._unpack_embeddings(sample, W, V)
        return self.loglikelihood(W, V, data)


    def logprob(self, Y, **kwargs):
        Mu = np.matmul(model.W[None], np.transpose(model.V, [0,2,1])).transpose([1,0,2])
        return norm.logpdf(Y, Mu, scale=np.sqrt(self.sigma2))


def _make_shared(arr, name):
    brr = sa.create('shm://{}'.format(name), arr.shape, dtype=arr.dtype)
    brr[:] = arr
    return brr

class MultiprocessingContext:
    def __init__(self, model):
        self.multiprocessing = True
        self.sharedprefix = model.sharedprefix
        self.gass_ngrid = model.gass_ngrid
        self.loglikelihood = model.loglikelihood
        self.has_ep_approx = model.Mu_ep is not None
        self.has_Row_constraints = model.Row_constraints is not None
        self.sigma2 = model.sigma2
        self.lam2 = model.lam2
        self.stability = model.stability

    def load(self):
        self.W = sa.attach(self.sharedprefix + 'W')
        self.V = sa.attach(self.sharedprefix + 'V')
        self.Tau2 = sa.attach(self.sharedprefix + 'Tau2')
        self.sigma2 = sa.attach(self.sharedprefix + 'sigma2')
        self.lam2 = sa.attach(self.sharedprefix + 'lam2')

        self.Constraints_A = sa.attach(self.sharedprefix + 'Constraints_A')
        self.Constraints_C = sa.attach(self.sharedprefix + 'Constraints_C')
        
        self.Delta = coo_matrix((sa.attach(self.sharedprefix + 'Delta_data'),
                                    (sa.attach(self.sharedprefix + 'Delta_row'), sa.attach(self.sharedprefix + 'Delta_col'))))

        self.Row_constraints = sa.attach(self.sharedprefix + 'Row_constraints') if self.has_Row_constraints else None
        self.Mu_ep = sa.attach(self.sharedprefix + 'Mu_ep') if self.has_ep_approx else None
        self.Sigma_ep = sa.attach(self.sharedprefix + 'Sigma_ep') if self.has_ep_approx else None

        self.nrows = self.W.shape[0]
        self.ncols, self.ndepth = self.V.shape[0], self.V.shape[1]
        self.nembeds = self.W.shape[1]
        self.nconstraints = self.Constraints_A.shape[0]
    
    

# Initialize the worker model
__worker_model = None
def _worker_init(model, user_fn):
    global __worker_model
    __worker_model = model
    model.load()
    if user_fn is not None:
        user_fn(model)
    
def _resample_W_i(args):
    i, data = args
    self = __worker_model

    # Enforce the lower-triangular structure of W
    ndims = min(self.nembeds, i+1)
    W_i = self.W[i,:ndims]
    V_i = self.V[:,:,:ndims]

    # Get the constraints given the current values of V
    Constraints = _w_constraints(self, i)
    
    # Update the mean and precision if we have an EP approximation
    if self.Mu_ep is not None:
        Mu_ep_i, Sigma_ep_i = self.Mu_ep[i], self.Sigma_ep[i]
        mu_i = ((Mu_ep_i / Sigma_ep_i**2)[...,None] * V_i).sum(axis=1).sum(axis=0)
        Q_i = ((V_i[:,:,:,None] / Sigma_ep_i[:,:,None,None]**2 * V_i[:,:,None]).sum(axis=1).sum(axis=0) + 
                    np.eye(ndims)/self.sigma2)
        mu_i = np.linalg.solve(Q_i, mu_i)
    else:
        mu_i = np.zeros(ndims)
        Q_i = np.eye(ndims)/self.sigma2
        Mu_ep_i, Sigma_ep_i = None, None

    # Sample a new W_i via generalized analytical slice sampling
    try:
        W_updated, _ = gass(W_i, Q_i, _w_loglikelihood, Constraints,
                         mu=mu_i,
                         ll_args=(i, data, V_i, Mu_ep_i, Sigma_ep_i, self.loglikelihood),
                         precision=True,
                         ngrid=self.gass_ngrid)
    except:
        np.set_printoptions(suppress=True, precision=2)
        print('W[{}]={}'.format(i, self.W[i]))
        print('V\n', self.V)
        print()
        print('Constraints A:\n', self.Constraints_A)
        if self.Row_constraints is not None:
            print('Row constraints and evaluation:')
            Row_constraints_and_eval = np.concatenate([self.Row_constraints, self.Row_constraints[:,:ndims].dot(W_i)[:,None]], axis=1)
            print(np.concatenate([Row_constraints_and_eval, (Row_constraints_and_eval[:,-1] >= Row_constraints_and_eval[:,-2])[:,None]],axis=1))
        print('Constraints W[{}]:\n{}'.format(i, Constraints))
        print()
        raise Exception()
    self.W[i,:ndims] = W_updated

def _w_constraints(self, i):
    # For tau_ij{1,...,t} effects, the constraints imply:
    # D<w_i,v_j{1,...,t}> = <Dv_j., w_i> > c
    # Enforce the lower-triangular structure of W
    ndims = min(self.nembeds, i+1)
    A = (self.Constraints_A[None,:,:,None] * self.V[:,None])[...,:ndims].sum(axis=2) # J x ncols x nembeds
    C = np.tile(self.Constraints_C, (self.ncols, 1))
    A = A.reshape((-1, ndims))
    Constraints_i = np.concatenate([A, C], axis=1)

    # Add any fixed constraints
    if self.Row_constraints is not None:
        Row_constraints_i = np.concatenate([self.Row_constraints[:,:ndims], self.Row_constraints[:,-1:]], axis=1)
        Constraints_i = np.concatenate([Constraints_i, Row_constraints_i], axis=0)
    return Constraints_i

def _w_loglikelihood(w_i, ll_args):
    idx, data, V_i, mu_ep, sigma_ep, loglikelihood = ll_args

    # The GASS sampler may call this with a single vector w_i or a batch
    # of multiple candidate vectors.
    if len(w_i.shape) > 1:
        # Calculate the mean effects for each w_i in the batch
        tau = (V_i[None] * w_i[:,None,None]).sum(axis=-1)

        # Get the black box log-likelihood
        # We could require that it supports batching, but this is cleaner,
        # albeit a little slower since it's not vectorized.
        ll = np.array([loglikelihood(data, tau_i, w_ik, V_i, row=idx) for tau_i, w_ik in zip(tau, w_i)])

        # Renormalize by the EP approximation, if we had one
        if mu_ep is not None:
            ll -= norm.logpdf(tau, mu_ep[None], sigma_ep[None]).sum(axis=-1).sum(axis=-1)
        assert len(ll) == w_i.shape[0]
        assert len(ll.shape) == 1
    else:
        # Calculate the mean effect
        tau = (V_i * w_i[None,None]).sum(axis=-1)

        # Get the black box log-likelihood
        ll = loglikelihood(data, tau, w_i, V_i, row=idx)

        # Renormalize by the EP approximation, if we had one
        if mu_ep is not None:
            # Divide by the normal likelihood
            ll -= norm.logpdf(tau, mu_ep, sigma_ep).sum()
    return ll

def _resample_V_j(args):
    j, data = args
    self = __worker_model
        
    Constraints = _v_constraints(self)
    I_embed = eye(self.nembeds, format='csc')
    try:
        # Get the global-local shrinkage prior as a diagonal precision matrix
        lam_Tau = spdiags((1/ (self.lam2 * self.Tau2[j]).clip(self.stability, 1/self.stability)), 0, self.Tau2.shape[1], self.Tau2.shape[1], format='csc')
        DLD = self.Delta.T.tocsc().dot(lam_Tau).dot(self.Delta)
        Q_prior = kron(I_embed, DLD).tocsc()

        if self.Mu_ep is not None:
            # Vectorize the EP tensor and remove missing data
            Mu_flat = self.Mu_ep[:,j].flatten()
            missing = np.isnan(Mu_flat)
            Mu_flat = Mu_flat[~missing]
            Sigma_flat = spdiags((1/self.Sigma_ep[:,j].flatten()[~missing])**2, 0, Mu_flat.shape[0], Mu_flat.shape[0], format='csc')

            # Treat the W embeddings as covariates in a linear regression
            X = kron(self.W, eye(self.ndepth, format='csc'), format='csc')[~missing]
            Xt = X.T.dot(Sigma_flat).tocsc()
            Q_likelihood = Xt.dot(X)
            sparsity_pattern = missing
            mu_part = Xt.dot(Mu_flat)

            # Ridge regression
            # stabilizer = spdiags(np.full(Q_prior.shape[0],0.99), 0, Q_prior.shape[0], Q_prior.shape[0], format='csc')
            Q = Q_likelihood + Q_prior

            factor = cholesky(Q)
            mu = factor.solve_A(mu_part)
            mu_ep_j = self.Mu_ep[:,j]
            sigma_ep_j = self.Sigma_ep[:,j]
        else:
            Q = Q_prior #+ spdiags(1e-8, 0, Q_prior.shape[0], Q_prior.shape[0], format='csc')
            factor = cholesky(Q)
            mu = np.zeros(Q.shape[0])
            mu_ep_j, sigma_ep_j = None, None
        
        V_j = self.V[j].T.flatten()
        V_args = (j, data, self, mu_ep_j, sigma_ep_j)
        
        V_updated = gass(V_j, factor,
                         _v_loglikelihood,
                         Constraints, 
                         ll_args=V_args,
                         mu=mu, sparse=True, precision=True, ngrid=self.gass_ngrid,
                         chol_factor=True, Q_shape=Q.shape, verbose=False)[0].reshape((self.nembeds, self.ndepth)).T
    except:
        print('Bad cholesky!')
        np.set_printoptions(precision=3, suppress=True, linewidth=250, edgeitems=100000, threshold=100000)
        print()
        print(j)
        print('W:')
        print(self.W)
        print()
        print('V[{}]:'.format(j))
        print(self.V[j])
        print()
        print('Mu[:,{}]:'.format(j))
        Mu = (self.W[:,None] * self.V[None,j]).sum(axis=-1)
        print(Mu)
        print()
        print('Monotonicity in Mu[:,{}]:'.format(j))
        print(Mu[:,:-1]-Mu[:,1:])
        print()
        print('Constraints A:')
        print(self.Constraints_A)
        # print('Constraints V:')
        # print(Constraints)
        print()
        print('V_j (flattened):')
        print(V_j)
        print('lam tau:')
        print(lam_Tau.todense())
        print('delta.lamtau.delta:')
        print(np.linalg.inv(self.Delta.T.tocsc().dot(lam_Tau).dot(self.Delta).todense()))
        print('Q:')
        print(np.linalg.inv(Q.todense()))
        print()
        print('Q condition number:')
        print(np.linalg.cond(np.linalg.inv(Q.todense())))
        raise Exception()
    
    # Update the column with the new sample
    self.V[j] =  V_updated

def _v_constraints(self):
    # Derive the constraints for each V_j matrix based on the current W values.
    # For tau_ij{1,...,t} effects, the constraints imply:
    # D<w_i,v_j{1,...,t}> = <v_j., Dw_i> > c
    A = (self.Constraints_A[None,:,None,:] * self.W[:,None,:,None]).reshape((self.nrows*self.nconstraints, self.nembeds*self.ndepth))
    C = np.tile(self.Constraints_C, (self.nrows, 1))
    V_constraints = np.concatenate([A,C], axis=1)
    return V_constraints

def _v_loglikelihood(V_j, ll_args):
    idx, data, self, mu_ep, sigma_ep = ll_args
    # The GASS sampler may call this with a single vector V_j or a batch
    # of multiple candidate vectors.
    if len(V_j.shape) > 1:
        # Batch of T x D matrices
        V_j = np.transpose(V_j.reshape((-1, self.nembeds, self.ndepth)), [0,2,1])

        # Calculate the mean effects for each V_j in the batch
        tau = (V_j[:,None] * self.W[None,:,None]).sum(axis=-1)

        # Get the black box log-likelihood
        # We could require that it supports batching, but this is cleaner,
        # albeit a little slower since it's not vectorized.
        ll = np.array([self.loglikelihood(data, tau_i, self.W, V_jk, col=idx) for tau_i, V_jk in zip(tau, V_j)])

        # Renormalize by the EP approximation, if we had one
        if mu_ep is not None:
            ll -= norm.logpdf(tau, mu_ep[None], sigma_ep[None]).sum(axis=-1).sum(axis=-1)

        assert len(ll) == V_j.shape[0]
        assert len(ll.shape) == 1
    else:
        # Change from a vector to a T x D matrix
        V_j = V_j.reshape((self.nembeds, self.ndepth)).T

        # Calculate the mean effect
        tau = (V_j[None] * self.W[:,None]).sum(axis=-1)

        # Get the black box log-likelihood
        ll = self.loglikelihood(data, tau, self.W, V_j, col=idx)

        # Renormalize by the EP approximation, if we had one
        if mu_ep is not None:
            # Divide by the normal likelihood
            ll -= norm.logpdf(tau, mu_ep, sigma_ep).sum()
    return ll

class ConstrainedNonconjugateBayesianTensorFiltering(BayesianTensorFiltering):
    def __init__(self,
                 nrows, ncols, ndepth, # Data is N x M x T
                 loglikelihood, # Generic black box loglikelihood
                 Constraints, # J x T matrix of J contraints for every Tau_{ij} T-vector
                 ep_approx=None, # optional EP approximation for centering
                 nthreads=3, # Number of parallel worker processes to use (technically not threads right now)
                 gass_ngrid=100, # Number of discrete points to approximate the valid ellipse regions
                 Row_constraints=None, # Optional extra matrix of permanant row constraint vectors
                 multiprocessing=True, # If true, uses nthreads processes instead of threads (works better on AWS)
                 sharedprefix=None, # A unique prefix to prepend to all variable names
                 worker_init=None, # Optional initialization function for workers
                 **kwargs):
        super().__init__(nrows, ncols, ndepth, **kwargs)
        self.loglikelihood = loglikelihood
        
        # The constraints can be arbitrary linear inequalities
        self.Constraints_A, self.Constraints_C = Constraints[:,:-1], Constraints[:,-1:]
        self.nconstraints = self.Constraints_A.shape[0]
        self.nthreads = nthreads
        self.gass_ngrid = gass_ngrid

        # Row constraints that remain fixed, as opposed to the
        # Constraints which are based on the opposite embedding
        self.Row_constraints = Row_constraints

        # If we were provided with a gaussian approximation, use that to
        # center the proposal
        if ep_approx is None:
            self.Mu_ep, self.Sigma_ep = None, None
        else:
            self.Mu_ep, self.Sigma_ep = ep_approx

        # Are we doing multi-processing or multi-threading?
        self.multiprocessing = multiprocessing
        if self.multiprocessing:
            # Use a unique prefix to prepend to all shared variables
            if sharedprefix is None:
                import uuid
                sharedprefix = str(uuid.uuid4())[:8]
            self.sharedprefix = sharedprefix

            # Convert variables to shared memory objects
            self.W = _make_shared(self.W, self.sharedprefix + 'W')
            self.V = _make_shared(self.V, self.sharedprefix + 'V')
            self.Tau2 = _make_shared(self.Tau2, self.sharedprefix + 'Tau2')
            self.Constraints_A = _make_shared(self.Constraints_A, self.sharedprefix + 'Constraints_A')
            self.Constraints_C = _make_shared(self.Constraints_C, self.sharedprefix + 'Constraints_C')
            self.Delta_data = _make_shared(self.Delta.data, self.sharedprefix + 'Delta_data')
            self.Delta_row = _make_shared(self.Delta.row, self.sharedprefix + 'Delta_row')
            self.Delta_col = _make_shared(self.Delta.col, self.sharedprefix + 'Delta_col')
            if self.Row_constraints is not None:
                self.Row_constraints = _make_shared(self.Row_constraints, self.sharedprefix + 'Row_constraints')
            if ep_approx is not None:
                self.Mu_ep = _make_shared(self.Mu_ep, self.sharedprefix + 'Mu_ep')
                self.Sigma_ep = _make_shared(self.Sigma_ep, self.sharedprefix + 'Sigma_ep')

            # Create shared memory for scalars
            self.sigma2_shared = sa.create('shm://{}sigma2'.format(self.sharedprefix), 1)
            self.lam2_shared = sa.create('shm://{}lam2'.format(self.sharedprefix), 1)
            self.sigma2_shared[:] = self.sigma2
            self.lam2_shared[:] = self.lam2

            # Create a process pool
            self.executor = Pool(self.nthreads, initializer=_worker_init, initargs=(MultiprocessingContext(self),worker_init))
        else:
            __worker_model = self
            self.executor = futures.ThreadPoolExecutor(max_workers=self.nthreads)

    def shutdown(self):
        if self.multiprocessing:
            self.executor.close()
            sa.delete(self.sharedprefix + 'W')
            sa.delete(self.sharedprefix + 'V')
            sa.delete(self.sharedprefix + 'Tau2')
            sa.delete(self.sharedprefix + 'sigma2')
            sa.delete(self.sharedprefix + 'lam2')
            sa.delete(self.sharedprefix + 'Constraints_A')
            sa.delete(self.sharedprefix + 'Constraints_C')
            sa.delete(self.sharedprefix + 'Delta_data')
            sa.delete(self.sharedprefix + 'Delta_row')
            sa.delete(self.sharedprefix + 'Delta_col')
            if self.Row_constraints is not None:
                sa.delete(self.sharedprefix + 'Row_constraints')
            if self.Mu_ep is not None:
                sa.delete(self.sharedprefix + 'Mu_ep')
                sa.delete(self.sharedprefix + 'Sigma_ep')
        else:
            self.executor.shutdown()

    def _resample_W(self, data):
        # Sample each W_i independently (should be easier to satisfy the constraints)
        if self.multiprocessing:
            resample_args = [(i, data) for i in range(self.nrows)]
            self.executor.map(_resample_W_i, resample_args)
        else:
            resample_args = [(i, data) for i in range(self.nrows)]
            self.executor.map(_resample_W_i, resample_args)
    
    def _resample_V(self, data):
        # Get the constraints given the current values of W
        if self.multiprocessing:
            resample_args = [(j, data) for j in range(self.ncols)]
            self.executor.map(_resample_V_j, resample_args)
        else:
            resample_args = [(j, data) for j in range(self.ncols)]
            self.executor.map(_resample_V_j, resample_args)

    def logprob(self, data, **kwargs):
        # Calculate the mean effects for each V_j in the batch
        tau = (self.W[:,None,None] * self.V[None]).sum(axis=-1)
        return self.loglikelihood(data, tau, self.W, self.V, self.rowcol_args)
    
    def _resample_sigma2(self):
        super()._resample_sigma2()

        if self.multiprocessing:
            self.sigma2_shared[:] = self.sigma2

    def _resample_lam2(self):
        super()._resample_lam2()

        if self.multiprocessing:
            self.lam2_shared[:] = self.lam2









