'''
Example code demonstrating how to factorize a functional matrix with Gaussian
likelihoods.
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import poisson
from scipy.sparse import csc_matrix
from functionalmf.factor import BinomialBayesianTensorFiltering
from functionalmf.utils import mse, mae, tensor_nmf, ilogit

nrows = 11
ncols = 12
ndepth = 20
nembeds = 3
nreplicates = 10

def init_model(tf_order=2, lam2=0.1, sigma2=0.5, nu2=1):
    # Setup the lower bound inequalities
    return BinomialBayesianTensorFiltering(nrows, ncols, ndepth,
                                                          nembeds=nembeds, tf_order=tf_order,
                                                          sigma2_init=sigma2, nthreads=1,
                                                          lam2_init=lam2)

def create_wiggly_with_jumps(break_prob=0.3):
    W = np.random.normal(0,1,size=(nrows, nembeds))
    if nrows > 1:
        W[np.triu_indices(nembeds, k=1)] = 0
    # V = np.random.gamma(1,1,size=(ncols, ndepth, nembeds)).cumsum(axis=1)[:,::-1,:]
    V = np.zeros((ncols, ndepth, nembeds))
    for j in range(ncols):
        x = np.random.normal(0,1,size=nembeds)
        coef = np.random.normal(0,0.1)
        V[j,-1] = x
        for k in range(ndepth-2, -1, -1):
            V[j,k] = V[j,k+1]
            if np.random.random() < break_prob:
                coef = np.random.normal(0,0.1)
                x = np.random.normal(0,1,size=nembeds)
            V[j,k] += coef*x
    return W, V

if __name__ == '__main__':
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    nburn = 10000
    nthin = 10
    nsamples = 1000

    np.random.seed(seed)

    # Sample from the prior
    model = init_model()

    # Ground truth drawn from the model
    W_true, V_true = create_wiggly_with_jumps()

    # Get the true mean values
    Mu = np.einsum('nk,mtk->nmt', W_true, V_true)
    print('Mean ranges: [{},{}]'.format(Mu.min(), Mu.max()))

    # Generate the data
    N = np.full((nrows, ncols, ndepth), nreplicates)
    Y = np.random.binomial(N, ilogit(Mu))

    # Convert to acceptable formats for sampling
    N = N.astype(float)
    Y = Y.astype(float)

    # Hold out some curves
    Y_missing = Y.copy()
    Y_missing[:3,:3] = np.nan
    N_missing = N.copy()
    N_missing[np.isnan(Y_missing)] = np.nan

    ####### Run the Gibbs sampler #######
    results = model.run_gibbs((Y_missing, N_missing), nburn=nburn, nthin=nthin, nsamples=nsamples, print_freq=50, verbose=True)
    Ws = results['W']
    Vs = results['V']
    Tau2s = results['Tau2']
    lam2s = results['lam2']
    sigma2s = results['sigma2']


    # Get the Bayes estimate
    Mu_hat = np.einsum('znk,zmtk->znmt', Ws, Vs)
    Mu_hat_mean = Mu_hat.mean(axis=0)
    Mu_hat_upper = np.percentile(Mu_hat, 95, axis=0)
    Mu_hat_lower = np.percentile(Mu_hat, 5, axis=0)
    
    ###### Plot the true curves, the noisy observations, and the fits ######
    print('Plotting results')
    X = np.arange(ndepth)
    fig, axarr = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows), sharex=True, sharey=True)
    for i in range(nrows):
        for j in range(ncols):
            ax = axarr[i,j]
            ax.plot(X, ilogit(Mu[i,j]), color='black')
            if len(Y.shape) > 3:
                for k in range(ndepth):
                    ax.scatter(np.full(Y.shape[-1],X[k]), Y[i,j,k] / N[i,j,k], color='gray')
            else:
                ax.scatter(X, Y[i,j] / N[i,j], color='gray')
            ax.plot(X, ilogit(Mu_hat_mean[i,j]), color='orange')
            ax.fill_between(X, ilogit(Mu_hat_lower[i,j]), ilogit(Mu_hat_upper[i,j]), color='orange', alpha=0.5)
    plt.ylim([np.nanmin(Y / N)-0.01, np.nanmax(Y / N)+0.01])
    plotdir = 'plots/'
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    plt.savefig(os.path.join(plotdir, 'binomial-tensor-filtering.pdf'), bbox_inches='tight')
    plt.close()






