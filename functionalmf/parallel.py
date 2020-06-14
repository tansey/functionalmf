import numpy as np
from functionalmf.factor import BayesianTensorFiltering
from multiprocessing import Pool
import scipy as sp
import warnings
from scipy.sparse import kron, eye, spdiags, csc_matrix, block_diag, diags
from scipy.stats import norm, multivariate_normal as mvn, gamma
from scipy.special import gammaln
from scipy.linalg import solve_triangular
from sksparse.cholmod import cholesky
from functionalmf.fast_mvn import sample_mvn_from_precision
from functionalmf.gass import gass
import SharedArray as sa

def _make_shared(arr, name):
    brr = sa.create('shm://{}'.format(name), arr.shape)
    brr[:] = arr
    return brr

class ParallelConstrainedNonconjugateBayesianTensorFiltering(BayesianTensorFiltering):
    def __init__(self,
                 nrows, ncols, ndepth, # Data is N x M x T
                 loglikelihood, # Generic black box loglikelihood
                 Constraints, # J x T matrix of J contraints for every Tau_{ij} T-vector
                 ep_approx=None, # optional EP approximation for centering
                 nthreads=3, # Number of parallel worker processes to use (technically not threads right now)
                 gass_ngrid=100, # Number of discrete points to approximate the valid ellipse regions
                 Row_constraints=None, # Optional extra matrix of permanant row constraint vectors
                 Col_constraints=None, # Optional extra matrix of permanent column constraint vectors
                 prefix=None, # Unique prefix string used for all shared memory allocation
                 **kwargs):
        super().__init__(nrows, ncols, ndepth, **kwargs)
        if prefix is None:
            import uuid
            prefix = uuid.uuid4()
        self.prefix = prefix
        self.loglikelihood = loglikelihood
        
        # The constraints can be arbitrary linear inequalities
        self.Constraints_A, self.Constraints_C = Constraints[:,:-1], Constraints[:,-1:]
        self.nconstraints = self.Constraints_A.shape[0]
        self.nthreads = nthreads
        self.gass_ngrid = gass_ngrid

        # Row and column constraints that remain fixed, as opposed to the
        # Constraints which are based on the opposite embedding
        self.Row_constraints = Row_constraints
        self.Col_constraints = Col_constraints

        self.rowcol_args = rowcol_args

        # If we were provided with a gaussian approximation, use that to
        # center the proposal
        if ep_approx is None:
            self.Mu_ep, self.Sigma_ep = None, None
        else:
            self.Mu_ep, self.Sigma_ep = ep_approx

        self.executor = Pool(self.nthreads)

        # Create the shared memory spaces and copy local memory over
        self.W = _make_shared(self.W, 'W')
        self.V = _make_shared(self.V, 'V')
        self.Constraints_A = _make_shared(self.Constraints_A, 'Constraints_A')
        self.Constraints_C = _make_shared(self.Constraints_C, 'Constraints_C')
        self.Tau2 = _make_shared(self.Tau2, 'Tau2')
        if ep_approx is not None:
            self.Mu_ep = _make_shared(self.Mu_ep, 'Mu_ep')
            self.Sigma_ep = _make_shared(self.Sigma_ep, 'Sigma_ep')
        if Row_constraints is not None:
            self.Row_constraints = _make_shared(self.Row_constraints, 'Row_constraints')


    def _resample_W(self, data):
        # Get the constraints given the current values of V
        Constraints = self._w_constraints()

        # Sample each W_i independently (should be easier to satisfy the constraints)
        if self.multiprocessing:
            resample_args = [(self, i, data, Constraints) for i in range(self.nrows)]
            ex = self.executor
            self.executor = None
            results = ex.map(_resample_W_worker, resample_args)
            self.executor = ex
        else:
            resample_args = [(i, data, Constraints) for i in range(self.nrows)]
            results = self.executor.map(lambda p: self._resample_W_i(*p), resample_args)
        for i,r in enumerate(results):
            ndims = min(self.nembeds, i+1)
            self.W[i,:ndims] = r

    def _resample_W_i(self, i, data, Constraints):
        # Enforce the lower-triangular structure of W
        ndims = min(self.nembeds, i+1)
        W_i = self.W[i,:ndims]
        V_i = self.V[:,:,:ndims]
        
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
            W_updated, _ = gass(W_i, Q_i, self._w_loglikelihood, Constraints[i],
                             mu=mu_i,
                             ll_args=(i, data, V_i, Mu_ep_i, Sigma_ep_i),
                             precision=True,
                             ngrid=self.gass_ngrid)
        except:
            np.set_printoptions(suppress=True, precision=2)
            print(i)
            print('W:')
            print(self.W)
            print()
            print('V')
            print(self.V)
            print()
            print('Constraints A:')
            print(self.Constraints_A)
            if self.Row_constraints is not None:
                print('Row constraints and evaluation:')
                Row_constraints_and_eval = np.concatenate([self.Row_constraints, self.Row_constraints[:,:-1].dot(self.W[i])[:,None]], axis=1)
                print(np.concatenate([Row_constraints_and_eval, (Row_constraints_and_eval[:,-1] >= Row_constraints_and_eval[:,-2])[:,None]],axis=1))
            print('Constraints W[{}]:'.format(i))
            print(Constraints[i])
            print()
            raise Exception()
        return W_updated
        
    def _resample_V(self, data):
        # Get the constraints given the current values of W
        Constraints = self._v_constraints()
        if self.multiprocessing:
            resample_args = [(self, j, data, Constraints) for j in range(self.ncols)]
            ex = self.executor
            self.executor = None
            results = ex.map(_resample_V_worker, resample_args)
            self.executor = ex
        else:
            resample_args = [(j, data, Constraints) for j in range(self.ncols)]
            results = self.executor.map(lambda p: self._resample_V_j(*p), resample_args)
        for j,r in enumerate(results):
            self.V[j] = r

    def _resample_V_j(self, j, data, Constraints):
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
            V_args = (j, data, mu_ep_j, sigma_ep_j)
            
            V_updated = gass(V_j, factor,
                             self._v_loglikelihood,
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
        return V_updated


    def _w_constraints(self):
        # Derive the constraints for each W_i vector based on the current V values.
        Constraints_W = []
        for i in range(self.nrows):
            # For tau_ij{1,...,t} effects, the constraints imply:
            # D<w_i,v_j{1,...,t}> = <Dv_j., w_i> > c
            if i < self.nembeds:
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
            Constraints_W.append(Constraints_i)
        return Constraints_W

    def _v_constraints(self):
        # Derive the constraints for each V_j matrix based on the current W values.
        # For tau_ij{1,...,t} effects, the constraints imply:
        # D<w_i,v_j{1,...,t}> = <v_j., Dw_i> > c
        A = (self.Constraints_A[None,:,None,:] * self.W[:,None,:,None]).reshape((self.nrows*self.nconstraints, self.nembeds*self.ndepth))
        C = np.tile(self.Constraints_C, (self.nrows, 1))
        V_constraints = np.concatenate([A,C], axis=1)
        if self.Col_constraints is not None:
            # TODO: enable constraints on the column embeddings
            raise NotImplementedError
        return V_constraints

    def _w_loglikelihood(self, w_i, ll_args):
        idx, data, V_i, mu_ep, sigma_ep = ll_args

        # The GASS sampler may call this with a single vector w_i or a batch
        # of multiple candidate vectors.
        if len(w_i.shape) > 1:
            # Calculate the mean effects for each w_i in the batch
            tau = (V_i[None] * w_i[:,None,None]).sum(axis=-1)

            # Get the black box log-likelihood
            # We could require that it supports batching, but this is cleaner,
            # albeit a little slower since it's not vectorized.
            ll = np.array([self.loglikelihood(data, tau_i, w_ik, V_i, self.rowcol_args, row=idx) for tau_i, w_ik in zip(tau, w_i)])

            # Renormalize by the EP approximation, if we had one
            if mu_ep is not None:
                ll -= norm.logpdf(tau, mu_ep[None], sigma_ep[None]).sum(axis=-1).sum(axis=-1)
            assert len(ll) == w_i.shape[0]
            assert len(ll.shape) == 1
        else:
            # Calculate the mean effect
            tau = (V_i * w_i[None,None]).sum(axis=-1)

            # Get the black box log-likelihood
            ll = self.loglikelihood(data, tau, w_i, V_i, self.rowcol_args, row=idx)

            # Renormalize by the EP approximation, if we had one
            if mu_ep is not None:
                # Divide by the normal likelihood
                # if len(w_i) == 1:
                #     print('ll: {}'.format(ll))
                #     print('normalizer: {}'.format(norm.logpdf(tau, mu_ep, sigma_ep).sum()))
                #     print('tau: {}'.format(tau))
                #     print('mu_ep: {}'.format(mu_ep))
                #     print('sigma_ep: {}'.format(sigma_ep))
                ll -= norm.logpdf(tau, mu_ep, sigma_ep).sum()
        return ll

    def _v_loglikelihood(self, V_j, ll_args):
        idx, data, mu_ep, sigma_ep = ll_args

        # The GASS sampler may call this with a single vector V_j or a batch
        # of multiple candidate vectors.
        if len(V_j.shape) > 1:
            # Batch of T x D matrices
            V_j = np.transpose(V_j.reshape((-1, self.nembeds, self.ndepth)), [0,2,1])
            # if idx == 0:
            #     print('Batch V_j:')
            #     print(V_j)

            # Calculate the mean effects for each V_j in the batch
            tau = (V_j[:,None] * self.W[None,:,None]).sum(axis=-1)

            # Get the black box log-likelihood
            # We could require that it supports batching, but this is cleaner,
            # albeit a little slower since it's not vectorized.
            ll = np.array([self.loglikelihood(data, tau_i, self.W, V_jk, self.rowcol_args, col=idx) for tau_i, V_jk in zip(tau, V_j)])

            # Renormalize by the EP approximation, if we had one
            if mu_ep is not None:
                ll -= norm.logpdf(tau, mu_ep[None], sigma_ep[None]).sum(axis=-1).sum(axis=-1)

            # if idx == 0:
            #     print('best proposal:')
            #     print(tau[np.argmax(ll)])
            #     print('ll: {}'.format(ll.max()))
            assert len(ll) == V_j.shape[0]
            assert len(ll.shape) == 1
        else:
            # Change from a vector to a T x D matrix
            V_j = V_j.reshape((self.nembeds, self.ndepth)).T

            # if idx == 0:
            #     print('Proposal V_j:')
            #     print(V_j)

            # Calculate the mean effect
            tau = (V_j[None] * self.W[:,None]).sum(axis=-1)
            # if idx == 0:
            #     print(tau)

            # Get the black box log-likelihood
            ll = self.loglikelihood(data, tau, self.W, V_j, self.rowcol_args, col=idx)

            # Renormalize by the EP approximation, if we had one
            if mu_ep is not None:
                # Divide by the normal likelihood
                # if len(V_j) == 1:
                #     print('ll: {}'.format(ll))
                #     print('normalizer: {}'.format(norm.logpdf(tau, mu_ep, sigma_ep).sum()))
                #     print('tau: {}'.format(tau))
                #     print('mu_ep: {}'.format(mu_ep))
                #     print('sigma_ep: {}'.format(sigma_ep))
                ll -= norm.logpdf(tau, mu_ep, sigma_ep).sum()
        return ll

    def logprob(self, data, **kwargs):
        # Calculate the mean effects for each V_j in the batch
        tau = (self.W[:,None,None] * self.V[None]).sum(axis=-1)
        return self.loglikelihood(data, tau, self.W, self.V, self.rowcol_args)
    

