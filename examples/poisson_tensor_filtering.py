'''
Example code demonstrating how to factorize a functional matrix with Poisson
likelihoods.
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import poisson
from scipy.sparse import csc_matrix
from functionalmf.factor import ConstrainedNonconjugateBayesianTensorFiltering,\
 GaussianBayesianTensorFiltering, NegativeBinomialBayesianTensorFiltering
from functionalmf.utils import mse, mae, tensor_nmf, ep_from_mf, ilogit, pav

nrows = 11
ncols = 12
ndepth = 20
nreplicates = 1

def coverage_at(truth, Mu, interval):
    lower = np.percentile(Mu, (100-interval)/2, axis=0)
    upper = np.percentile(Mu, (100-interval)/2 + interval, axis=0)
    return np.mean((truth >= lower) & (truth <= upper)) * 100

def rowcol_loglikelihood(Y, WV, W, V, row=None, col=None):
    if row is not None:
        Y = Y[row]
    if col is not None:
        Y = Y[:,col]
    if len(Y.shape) > len(WV.shape):
        WV = WV[...,None]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        z = np.nansum(poisson.logpmf(Y, WV))
    return z

def init_model(nembeds, tf_order=0, lam2=0.1, sigma2=0.5, sharedprefix='poisson-btf', monotone=False, nthreads=3):
    clean_up(sharedprefix)

    # Constraints requiring positive means
    Constraints = np.concatenate([np.eye(ndepth), np.zeros((ndepth,1))], axis=1)

    # Constraints requiring monotone decreasing
    if monotone:
        C_mono = np.array([np.concatenate([np.zeros(i), [1,-1], np.zeros(ndepth-i-2), [-1e-2]]) for i in range(ndepth-1)])
        Constraints = np.concatenate([Constraints, C_mono], axis=0)
    
    # Setup the lower bound inequalities
    return ConstrainedNonconjugateBayesianTensorFiltering(nrows, ncols, ndepth,
                                                          rowcol_loglikelihood,
                                                          Constraints,
                                                          nembeds=nembeds, tf_order=tf_order,
                                                          sigma2_init=sigma2, nthreads=nthreads,
                                                          lam2_init=lam2, multiprocessing=True,
                                                          sharedprefix=sharedprefix)

def setup_sampler(model, Y, monotone=False):
    # Pick which variables to sample and which to fix at the truth
    model.sample_W = True
    model.sample_V = True
    model.sample_Tau2 = True
    model.sample_sigma2 = True
    model.sample_lam2 = True

    # Use nonnegative matrix factorization to initialize
    if model.sample_W and model.sample_V:
        nmf_W, nmf_V = tensor_nmf(Y, model.nembeds, monotone=monotone)
        model.W[:] = nmf_W
        model.V[:] = nmf_V
        # model.Mu_ep, model.Sigma_ep = ep_from_mf(Y, model.W, model.V, mode='multiplier', multiplier=3)


    if model.sample_lam2:
        model._init_lam2()

    if model.sample_Tau2:
        model._init_Tau2()

    if model.sample_sigma2:
        model._init_sigma2()

def create_piecewise_constant(break_prob=0.2, ndims=3):
    W = np.random.gamma(1,1,size=(nrows, ndims))
    if nrows > 1:
        W[np.triu_indices(ndims, k=1)] = 0
    # V = np.random.gamma(1,1,size=(ncols, ndepth, nembeds)).cumsum(axis=1)[:,::-1,:]
    V = np.zeros((ncols, ndepth, ndims))
    for j in range(ncols):
        V[j,-1] = np.random.gamma(1,1,size=ndims)
        for k in range(ndepth-2, -1, -1):
            V[j,k] = V[j,k+1]
            if np.random.random() < break_prob:
                V[j,k] += np.random.gamma(1,1,size=ndims)
    return W, V

def try_delete(name):
    import SharedArray as sa
    try:
        sa.delete(name)
    except:
        pass

def clean_up(sharedprefix):
    # In case we exited early, clean up stuff -- TOOD: make this automatic in BTF
    try_delete(sharedprefix + 'X')
    try_delete(sharedprefix + 'U')
    try_delete(sharedprefix + 'Y_obs')
    try_delete(sharedprefix + 'W')
    try_delete(sharedprefix + 'V')
    try_delete(sharedprefix + 'sigma2')
    try_delete(sharedprefix + 'lam2')
    try_delete(sharedprefix + 'Tau2')
    try_delete(sharedprefix + 'Constraints_A')
    try_delete(sharedprefix + 'Constraints_C')
    try_delete(sharedprefix + 'Row_constraints')
    try_delete(sharedprefix + 'Mu_ep')
    try_delete(sharedprefix + 'Sigma_ep')
    try_delete(sharedprefix + 'Delta_data')
    try_delete(sharedprefix + 'Delta_row')
    try_delete(sharedprefix + 'Delta_col')

if __name__ == '__main__':
    import sys
    nburn = 50
    nthin = 5
    nsamples = 10
    nchains = 1
    nembeds_options = [2,3,5,10]
    nthreads = 1 if len(sys.argv) == 1 else int(sys.argv[1])
    
    aggregate_performance = {nembeds: [] for nembeds in nembeds_options}
    for seed in [1,2,3,4,5]:
        np.random.seed(seed)

        # Ground truth drawn from the model
        W_true, V_true = create_piecewise_constant()

        # Get the true mean values
        Mu = np.einsum('nk,mtk->nmt', W_true, V_true)

        # Generate the data
        Y = np.random.poisson(Mu[...,None], size=(nrows, ncols, ndepth, nreplicates)).astype(float)

        # Hold out some curves
        Y_missing = Y.copy()
        Y_missing[:3,:3] = np.nan

        for nembeds in nembeds_options:
            print('Seed {} d={}'.format(seed, nembeds))
            models = []

            ############### Setup the NMF baseline ###############
            W_nmf, V_nmf = tensor_nmf(Y_missing, nembeds)
            Mu_nmf = (W_nmf[:,None,None] * V_nmf[None]).sum(axis=-1)
            models.append({'name': 'NMF', 'fit': Mu_nmf, 'samples': Mu_nmf[None], 'file': 'nmf.npy'})
            ###########################################################################

            ############### Setup the PGDS baseline ###############
            print('Fitting PGDS')
            # try:
            for tau in [0.25, 0.5, 1]:
                # If you have the Poisson-gamma dynamical system of Schein et al installed,
                # add that baseline comparison
                # sys.path.append('../apf/src/')
                from functionalmf.pgds import fit_pgds
                # Fit the PGDS model
                print('\tk={} tau={}'.format(nembeds, tau))
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    Mu_pgds, (W_pgds, V_pgds, U_pgds) = fit_pgds(Y_missing.sum(axis=-1), nembeds, nburn=nburn, nthin=nthin, nsamples=nsamples, tau=tau, nthreads=1)
                Mu_pgds_mean = Mu_pgds.mean(axis=0) / Y_missing.shape[-1]
                models.append({'name': 'PGDS tau={}'.format(tau), 'fit': Mu_pgds_mean, 'samples': Mu_pgds, 'file': 'pgds_{}.npy'.format(tau)})
            # except:
            #     print('Could not run PGDS, bailing.')
            #     Mu_pgds_mean = Y_missing
            #     pass

            ############### Setup the Negative Binomial model ###############
            tf_order = 0
            nbinom_model = NegativeBinomialBayesianTensorFiltering(nrows, ncols, ndepth,
                                                                      nembeds=nembeds, tf_order=tf_order,
                                                                      sigma2_init=1, nthreads=nthreads,
                                                                      lam2_init=0.1, nu2_init=1)
            results = nbinom_model.run_gibbs(Y_missing, nburn=nburn, nthin=nthin, nsamples=nsamples, print_freq=1000, verbose=True)
            Ws = results['W']
            Vs = results['V']
            Rs = results['R']
            Tau2s = results['Tau2']
            lam2s = results['lam2']
            sigma2s = results['sigma2']

            # Get the Bayes estimate
            Ps = ilogit(np.einsum('znk,zmtk->znmt', Ws, Vs).clip(-10,10))
            Mu_nbinom = Rs * Ps / (1 - Ps)
            Mu_nbinom_mean = Mu_nbinom.mean(axis=0)
            Mu_nbinom_upper = np.percentile(Mu_nbinom, 95, axis=0)
            Mu_nbinom_lower = np.percentile(Mu_nbinom, 5, axis=0)
            models.append({'name': 'NB-BTF', 'fit': Mu_nbinom_mean, 'samples': Mu_nbinom, 'file': 'btf_nbinom.npy'})
            ###########################################################################

            ############### Setup the Poisson BTF model ###############
            # Sample from the prior
            model = init_model(nembeds, sharedprefix='pbtf-{}-{}'.format(seed, nembeds), nthreads=nthreads)

            # Reinitialize the parts we're going to sample
            setup_sampler(model, Y_missing)

            # Save the initial model values
            X = np.arange(ndepth)
            Mu_init = np.matmul(model.W[None], np.transpose(model.V, [0,2,1])).transpose([1,0,2])

            ####### Run the Gibbs sampler #######
            results = model.run_gibbs(Y_missing, nburn=nburn, nthin=nthin, nsamples=nsamples, print_freq=1000, verbose=True)
            for chain in range(nchains-1):
                print('Chain {}'.format(chain+2))
                setup_sampler(model, Y)
                chain_results = model.run_gibbs(Y_missing, nburn=nburn, nthin=nthin, nsamples=nsamples)
                for key, val in chain_results.items():
                    results[key] = np.concatenate([results[key], val], axis=0)
            Ws = results['W']
            Vs = results['V']
            Tau2s = results['Tau2']
            lam2s = results['lam2']
            sigma2s = results['sigma2']

            # Get the Bayes estimate
            Mu_hat = np.matmul(Ws[:,None], np.transpose(Vs, [0,1,3,2])).transpose([0,2,1,3])
            Mu_hat_mean = Mu_hat.mean(axis=0)
            Mu_hat_upper = np.percentile(Mu_hat, 95, axis=0)
            Mu_hat_lower = np.percentile(Mu_hat, 5, axis=0)
            models.append({'name': 'Poisson-BTF', 'fit': Mu_hat_mean, 'samples': Mu_hat, 'file': 'btf_poisson.npy'})
            ###########################################################################
            

            metrics = [#{'name': 'MAE (all data)',  'fun': lambda Y, Mu, pred, samples: mae(Y, pred[...,None])},
                       #{'name': 'RMSE (all data)', 'fun': lambda Y, Mu, pred, samples: np.sqrt(mse(Y, pred[...,None]))},
                       #{'name': 'NLL (all data)', 'fun': lambda Y, Mu, pred, samples: -np.nansum(poisson.logpmf(Y, pred[...,None]))},
                       {'name': 'MAE (held out)',  'fun': lambda Y, Mu, pred, samples: mae(Y[:3,:3], pred[:3,:3,...,None])},
                       {'name': 'RMSE (held out)', 'fun': lambda Y, Mu, pred, samples: np.sqrt(mse(Y[:3,:3], pred[:3,:3,...,None]))},
                       {'name': 'NLL (held out)', 'fun': lambda Y, Mu, pred, samples: -np.nansum(poisson.logpmf(Y[:3,:3], pred[:3,:3,...,None]))},
                       {'name': 'MAE (true rate)', 'fun': lambda Y, Mu, pred, samples: mae(Mu, pred)},
                       {'name': 'RMSE (true rate)', 'fun': lambda Y, Mu, pred, samples: np.sqrt(mse(Mu, pred))},
                       {'name': '50% Coverage', 'fun': lambda Y, Mu, pred, samples: coverage_at(Mu, samples, 50)},
                       {'name': '75% Coverage', 'fun': lambda Y, Mu, pred, samples: coverage_at(Mu, samples, 75)},
                       {'name': '90% Coverage', 'fun': lambda Y, Mu, pred, samples: coverage_at(Mu, samples, 90)},
                       {'name': '95% Coverage', 'fun': lambda Y, Mu, pred, samples: coverage_at(Mu, samples, 95)}]
            nmetrics = len(metrics)
            nmodels = len(models)

            metric_results = np.zeros((nmetrics, nmodels))
            for metidx, metric in enumerate(metrics):
                metric_results[metidx] = [metric['fun'](Y, Mu, m['fit'], m['samples']) for m in models]

            print('Saving results to file')
            outdir = os.path.join('data/poisson_tensor_filtering/', 'seed{}-nembeds{}'.format(seed, nembeds))
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            np.save(os.path.join(outdir, 'y'), Y)
            np.save(os.path.join(outdir, 'mu'), Mu)
            [np.save(os.path.join(outdir, m['file'].replace('.npy', '_fit.npy')), m['fit']) for m in models]
            [np.save(os.path.join(outdir, m['file'].replace('.npy', '_samples.npy')), m['samples']) for m in models]
            print()

            model.shutdown()

            aggregate_performance[nembeds].append(metric_results)

    for nembeds in nembeds_options:
        print('d={}'.format(nembeds))
        nembeds_results = np.array(aggregate_performance[nembeds]).mean(axis=0)
        print(('{:<18}'*(nmetrics+1)).format(*(['Model'] + [m['name'] for m in metrics])))
        for i, m in enumerate(models):
            model_results = ''.join(['{:<18.2f}'.format(r) for r in nembeds_results[:,i]])
            print('{:<18}'.format(m['name']) + model_results)

    ###### Plot the true curves, the noisy observations, and the fits ######
    # print('Plotting results')
    # X = np.arange(ndepth)
    # fig, axarr = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows), sharex=True, sharey=True)
    # for i in range(nrows):
    #     for j in range(ncols):
    #         ax = axarr[i,j]
    #         ax.plot(X, Mu[i,j], color='black')
    #         ax.plot(X, Mu_init[i,j], color='blue')
    #         if model.Mu_ep is not None:
    #             ax.errorbar(X, model.Mu_ep[i,j], yerr=model.Sigma_ep[i,j], label='EP approximation', alpha=0.5)
    #         if len(Y.shape) > 3:
    #             for k in range(ndepth):
    #                 ax.scatter(np.full(Y.shape[-1],X[k]), Y[i,j,k], color='gray')
    #         else:
    #             ax.scatter(X, Y[i,j], color='gray')
    #         ax.plot(X, Mu_hat_mean[i,j], color='orange')
    #         ax.fill_between(X, Mu_hat_lower[i,j], Mu_hat_upper[i,j], color='orange', alpha=0.5)
    # plt.ylim([0, np.nanmax(Y)+0.01])
    # plotdir = 'plots/'
    # if not os.path.exists(plotdir):
    #     os.makedirs(plotdir)
    # plt.savefig(os.path.join(plotdir, 'poisson-tensor-filtering.pdf'), bbox_inches='tight')
    # plt.close()









