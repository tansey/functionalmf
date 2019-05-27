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
from functionalmf.factor import ConstrainedNonconjugateBayesianTensorFiltering
from functionalmf.utils import mse, mae, tensor_nmf

nrows = 11
ncols = 12
ndepth = 20
nembeds = 3
nreplicates = 1

def rowcol_loglikelihood(Y, WV, row=None, col=None):
    if row is not None:
        Y = Y[row]
    if col is not None:
        Y = Y[:,col]
    # missing = np.isnan(Y)
    if len(Y.shape) > len(WV.shape):
        WV = WV[...,None]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        z = np.nansum(poisson.logpmf(Y, WV))
    return z

def init_model(tf_order=0, lam2=0.1, sigma2=0.5):
    # Constraints requiring positive means
    C_zero = np.concatenate([np.eye(ndepth), np.zeros((ndepth,1))], axis=1)
    
    # Setup the lower bound inequalities
    return ConstrainedNonconjugateBayesianTensorFiltering(nrows, ncols, ndepth,
                                                          rowcol_loglikelihood,
                                                          C_zero,
                                                          nembeds=nembeds, tf_order=tf_order,
                                                          sigma2_init=sigma2, nthreads=1,
                                                          lam2_init=lam2)

def ep_from_nmf(Y, W, V):
    if len(Y.shape) == 3:
        Y = Y[...,None]
    M = (W[:,None,None] * V[None]).sum(axis=-1, keepdims=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sqerr = np.nanmean((Y - M)**2, axis=-1)
        overestimate = np.sqrt(np.nanmax(sqerr))
    print('Estimated stdev: {}'.format(overestimate))
    return M[...,0], np.ones(Y.shape[:-1])*overestimate

def setup_sampler(model, Y):
    # Pick which variables to sample and which to fix at the truth
    model.sample_W = True
    model.sample_V = True
    model.sample_Tau2 = True
    model.sample_sigma2 = True
    model.sample_lam2 = True

    # Use nonnegative matrix factorization to initialize
    if model.sample_W and model.sample_V:
        model.W, model.V = tensor_nmf(Y, nembeds)
        model.Mu_ep, model.Sigma_ep = ep_from_nmf(Y, model.W, model.V)

    if model.sample_lam2:
        model._init_lam2()

    if model.sample_Tau2:
        model._init_Tau2()

    if model.sample_sigma2:
        model._init_sigma2()

def create_piecewise_constant(break_prob=0.2):
    W = np.random.gamma(1,1,size=(nrows, nembeds))
    if nrows > 1:
        W[np.triu_indices(nembeds, k=1)] = 0
    # V = np.random.gamma(1,1,size=(ncols, ndepth, nembeds)).cumsum(axis=1)[:,::-1,:]
    V = np.zeros((ncols, ndepth, nembeds))
    for j in range(ncols):
        V[j,-1] = np.random.gamma(1,1,size=nembeds)
        for k in range(ndepth-2, -1, -1):
            V[j,k] = V[j,k+1]
            if np.random.random() < break_prob:
                V[j,k] += np.random.gamma(1,1,size=nembeds)
    return W, V

if __name__ == '__main__':
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    nburn = 2000
    nthin = 1
    nsamples = 2000
    nchains = 1

    np.random.seed(seed)

    # Sample from the prior
    model = init_model()

    # Ground truth drawn from the model
    W_true, V_true = create_piecewise_constant()

    # Get the true mean values
    Mu = np.einsum('nk,mtk->nmt', W_true, V_true)

    # Generate the data
    Y = np.random.poisson(Mu[...,None], size=(nrows, ncols, ndepth, nreplicates)).astype(float)

    # Hold out some curves
    Y_missing = Y.copy()
    Y_missing[:3,:3] = np.nan

    # Reinitialize the parts we're going to sample
    setup_sampler(model, Y_missing)

    # Save the initial model values
    X = np.arange(ndepth)
    Mu_init = np.matmul(model.W[None], np.transpose(model.V, [0,2,1])).transpose([1,0,2])

    ####### Run the Gibbs sampler #######
    results = model.run_gibbs(Y_missing, nburn=nburn, nthin=nthin, nsamples=nsamples, print_freq=1, verbose=True)
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

    # Get the raw NMF as a baseline
    W_nmf, V_nmf = tensor_nmf(Y_missing, nembeds)
    Mu_nmf = (W_nmf[:,None,None] * V_nmf[None]).sum(axis=-1)
    models = [{'name': 'NMF', 'fit': Mu_nmf, 'file': 'nmf.npy'}]

    try:
        # If you have the Poisson-gamma dynamical system of Schein et al installed,
        # add that baseline comparison
        sys.path.append('../apf/src/')
        from smoothbayes.pgds import fit_pgds
        # Fit the PGDS model
        print('Fitting to full data with k={}'.format(nembeds))
        Mu_pgds, (W_pgds, V_pgds, U_pgds) = fit_pgds(Y_missing[...,0], nembeds, nburn=2000, nthin=1, nsamples=2000)
        Mu_pgds_mean = Mu_pgds.mean(axis=0)
        models.append({'name': 'PGDS', 'fit': Mu_pgds_mean, 'file': 'pgds.npy'})
    except:
        pass

    # Get the Bayes estimate
    Mu_hat = np.matmul(Ws[:,None], np.transpose(Vs, [0,1,3,2])).transpose([0,2,1,3])
    Mu_hat_mean = Mu_hat.mean(axis=0)
    Mu_hat_upper = np.percentile(Mu_hat, 95, axis=0)
    Mu_hat_lower = np.percentile(Mu_hat, 5, axis=0)
    models.append({'name': 'BTF', 'fit': Mu_hat_mean, 'file': 'btf.npy'})

    metrics = [{'name': 'MAE (all data)',  'fun': lambda Y, Mu, pred: mae(Y, pred[...,None])},
               {'name': 'RMSE (all data)', 'fun': lambda Y, Mu, pred: np.sqrt(mse(Y, pred[...,None]))},
               {'name': 'NLL (all data)', 'fun': lambda Y, Mu, pred: -np.nansum(poisson.logpmf(Y, pred[...,None]))},
               {'name': 'MAE (held out data)',  'fun': lambda Y, Mu, pred: mae(Y[:3,:3], pred[:3,:3,...,None])},
               {'name': 'RMSE (held out data)', 'fun': lambda Y, Mu, pred: np.sqrt(mse(Y[:3,:3], pred[:3,:3,...,None]))},
               {'name': 'NLL (all data)', 'fun': lambda Y, Mu, pred: -np.nansum(poisson.logpmf(Y[:3,:3], pred[:3,:3,...,None]))},
               {'name': 'MAE (true rate)', 'fun': lambda Y, Mu, pred: mae(Mu, pred)},
               {'name': 'RMSE (true rate)', 'fun': lambda Y, Mu, pred: np.sqrt(mse(Mu, pred))}]
    nmetrics = len(metrics)
    nmodels = len(models)

    metric_results = np.zeros((nmetrics, nmodels))
    for metidx, metric in enumerate(metrics):
        metric_results[metidx] = [metric['fun'](Y, Mu, m['fit']) for m in models]

    print('Results:')
    print(('{:<20}'*(nmetrics+1)).format(*(['Model'] + [m['name'] for m in metrics])))
    for i, m in enumerate(models):
        model_results = ''.join(['{:<20.2f}'.format(r) for r in metric_results[:,i]])
        print('{:<20}'.format(m['name']) + model_results)

    print('Saving results to file')
    outdir = os.path.join('data/poisson_tensor_filtering/', 'seed{}'.format(seed))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    np.save(os.path.join(outdir, 'y'), Y)
    np.save(os.path.join(outdir, 'mu'), Mu)
    [np.save(os.path.join(outdir, m['file']), m['fit']) for m in models]
    np.save(os.path.join(outdir, 'btf_ep_sigma'), model.Sigma_ep)
    print()

    ###### Plot the true curves, the noisy observations, and the fits ######
    print('Plotting results')
    X = np.arange(ndepth)
    fig, axarr = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows), sharex=True, sharey=True)
    for i in range(nrows):
        for j in range(ncols):
            ax = axarr[i,j]
            ax.plot(X, Mu[i,j], color='black')
            ax.plot(X, Mu_init[i,j], color='blue')
            if model.Mu_ep is not None:
                ax.errorbar(X, model.Mu_ep[i,j], yerr=model.Sigma_ep[i,j], label='EP approximation', alpha=0.5)
            if len(Y.shape) > 3:
                for k in range(ndepth):
                    ax.scatter(np.full(Y.shape[-1],X[k]), Y[i,j,k], color='gray')
            else:
                ax.scatter(X, Y[i,j], color='gray')
            ax.plot(X, Mu_hat_mean[i,j], color='orange')
            ax.fill_between(X, Mu_hat_lower[i,j], Mu_hat_upper[i,j], color='orange', alpha=0.5)
    plt.ylim([0, np.nanmax(Y)+0.01])
    plotdir = 'plots/'
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    plt.savefig(os.path.join(plotdir, 'poisson-tensor-filtering.pdf'), bbox_inches='tight')
    plt.close()



