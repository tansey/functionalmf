import abc
import numpy as np
import scipy as sp
from scipy.stats import invgauss, norm, multivariate_normal, binom
from scipy.sparse import csc_matrix, coo_matrix, issparse, eye as sparse_eye
from sksparse.cholmod import cholesky
from functionalmf.elliptical_slice import elliptical_slice_
from functionalmf.utils import ilogit
from functionalmf.fast_mvn import sample_mvn_from_precision
from pypolyagamma import PyPolyaGamma

class _BayesianModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def resample(self, data, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def _inferred_variables(self, var_map):
        raise NotImplementedError

    @abc.abstractmethod
    def _default_hyperparam_options(self, hyperparams, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def _set_hyperparameters(self, hyperparams):
        raise NotImplementedError

    @abc.abstractmethod
    def logprob(self, data, **kwargs):
        raise NotImplementedError

    def inferred_variables(self):
        '''All non-nuisance parameters inferred by calling resample.'''
        results = {}
        self._inferred_variables(results)
        return results

    def run_gibbs(self, data, nburn=1000, nthin=1, nsamples=1000, verbose=True, print_freq=100, **kwargs):
        '''Run a gibbs sampler on the model. TODO: multithreaded multichains'''
        nsteps = nburn + nthin*nsamples
        for step in range(nsteps):
            if verbose and step % print_freq == 0:
                print('\tStep {}'.format(step))
            # Gibbs step
            self.resample(data, **kwargs)

            # Save posterior samples after burn-in
            if step >= nburn and (step-nburn) % nthin == 0:
                sidx = (step - nburn) // nthin

                # Get the variables
                inferred = self.inferred_variables()

                # If this is the first step, build the results dictionary
                if sidx == 0:
                    results = {}
                    for key, val in inferred.items():
                        results[key] = np.zeros([nsamples] + ([1] if np.isscalar(val) else list(val.shape)))

                # Save all the sampled variables that may be of interest.
                for key, val in inferred.items():
                    results[key][sidx] = val
        return results
        

    def select_hyperparams_DIC(self, data, verbose=True, **kwargs):
        '''Selects a good value of the hyperparameters using the deviance
        information criterion (DIC):

        DIC = 2 * avg(dev(beta)) - dev(avg(beta))
        dev(beta_i) = -2 log(p(y | beta_i))

        '''
        import itertools
        hyperparam_options = {}
        self._default_hyperparam_options(hyperparam_options, **kwargs)

        if verbose:
            print('Grid search for hyperparameters:')
            for key, val in hyperparam_options.items():
                print('{}: {} values from {} to {}'.format(key, len(val), min(val), max(val)))

        # Convert the map to lists of names and option lists
        param_names = list(hyperparam_options.keys())
        param_options = [hyperparam_options[name] for name in param_names]

        # We use the deviance information criterion to select the best values
        all_indices = [d for d in np.ndindex(*[len(p) for p in param_options])]
        dic_scores = np.zeros(len(all_indices))
        best_results, best_score, best_idx = None, None, None

        for score_idx, indices in enumerate(all_indices):
            if verbose:
                print(' '.join(['{}={}'.format(param_names[pidx], param_options[pidx][val_idx]) for pidx, val_idx in enumerate(indices)]))
            
            # Create the hyperparams
            cur_options = {param_names[pidx]: param_options[pidx][val_idx] for pidx, val_idx in enumerate(indices)}

            # Set the hyperparameters
            self._set_hyperparameters(cur_options)

            # Run the gibbs sampler
            results = self.run_gibbs(data, verbose=False, **kwargs)
            
            # Number of posterior samples
            nsamples = next(iter(results.values())).shape[0]

            # Get the mean parameters
            mean_results = {key: val.mean(axis=0) for key, val in results.items()}

            # Calculate the DIC score:
            # dev(beta_i) = -2 log(p(y | beta_i))
            # DIC = 2 * avg(dev(beta)) - dev(avg(beta))
            D_mean = -2 * self.logprob(data, **mean_results)
            mean_D =  -2 * np.mean([self.logprob(data,
                                **{key: val[i] for key,val in results.items()})
                                 for i in range(nsamples)])
            dic_scores[score_idx] = 2*mean_D - D_mean

            # The best model is the one with lowest DIC
            if best_score is None or dic_scores[score_idx] < best_score: # TEMP
                best_results = results
                best_score = dic_scores[score_idx]
                best_idx = score_idx

        # Set parameters to the best values
        best_options = {param_names[pidx]: param_options[pidx][val_idx] for pidx, val_idx in enumerate(all_indices[best_idx])}
        self._set_hyperparameters(best_options)

        return {'scores': dic_scores,
                'options': hyperparam_options,
                'best': best_options,
                'fit': best_results}


class ConjugateInverseGammaPrior(object):
    '''
    Prior for a diagonal precision matrix in a multivariate normal likelihood
    model. Uses a shape and rate parameterization of the gamma.
    '''
    def __init__(self, N, shape=0.1, rate=0.1):
        self.N = N
        self.shape = shape
        self.rate = rate

    def resample(self, data, **kwargs):
        means, obs = data
        if np.isscalar(means):
            means = np.array([means])
        if np.isscalar(obs):
            obs = np.array([obs])

        missing = np.isnan(obs)

        # Calculate posterior hyperparams
        sqerr = np.nansum((means - obs)**2)
        a_post = self.shape + np.sum(~missing) / 2
        b_post = self.rate + sqerr / 2

        # Sample from the posterior
        # Numpy uses a scale parameterization
        sigma2_inv = np.random.gamma(a_post, 1/b_post)
        if self.N == 1:
            return sigma2_inv
        return np.full(self.N, sigma2_inv)

    def draw_from_prior(self, size=1):
        return np.random.gamma(self.shape, 1/self.rate, size=size)







