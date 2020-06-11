'''
Example code demonstrating how to factorize a functional matrix with Negative Binomial
likelihoods.
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import poisson
from functionalmf.factor import NegativeBinomialBayesianTensorFiltering
from functionalmf.utils import ilogit

nrows = 11
ncols = 12
ndepth = 20
nembeds = 3
nreplicates = 1

def init_model(tf_order=2, lam2=0.1, sigma2=0.5):
    # Setup the model
    return NegativeBinomialBayesianTensorFiltering(nrows, ncols, ndepth,
                                              nembeds=nembeds, tf_order=tf_order,
                                              sigma2_init=sigma2, nthreads=1,
                                              lam2_init=lam2,
                                              rdims=(1,2))

def create_wiggly_with_jumps(break_prob=0.3):
    W = np.random.normal(0,1,size=(nrows, nembeds))
    if nrows > 1:
        W[np.triu_indices(nembeds, k=1)] = 0
    V = np.zeros((ncols, ndepth, nembeds))
    for j in range(ncols):
        x = np.random.normal(0,1,size=nembeds)
        coef = np.random.normal(0,1)
        V[j,-1] = x
        for k in range(ndepth-2, -1, -1):
            V[j,k] = V[j,k+1]
            if np.random.random() < break_prob:
                coef = np.random.normal(0,1)
                x = np.random.normal(0,1,size=nembeds)
            V[j,k] += coef*x
    return W, V

def create_piecewise_constant(break_prob=0.2):
    W = np.random.gamma(1,1,size=(nrows, nembeds))
    if nrows > 1:
        W[np.triu_indices(nembeds, k=1)] = 0
    V = np.zeros((ncols, ndepth, nembeds))
    for j in range(ncols):
        V[j,-1] = np.random.gamma(1,1,size=nembeds)
        for k in range(ndepth-2, -1, -1):
            V[j,k] = V[j,k+1]
            if np.random.random() < break_prob:
                V[j,k] += np.random.gamma(1,1,size=nembeds)
    Mu = np.einsum('nk,mzk->nmz', W, V)
    Variance = np.random.gamma(1,scale=1,size=(nrows,1,1))*Mu**2 + Mu
    P = 1 - Mu / Variance
    R = Mu * (1-P) / P 
    return R, P, Mu, Variance
    # rp/1-p = mu, rp/(1-p)**2 = sigma
    # r = mu * (1-p) / p
    # sigma = mu / (1-p)
    # mu/sigma = 1-p => 1 - mu/sigma = p
    # (1-p) = mu/sigma ==> p = 1-mu/sigma

# if __name__ == '__main__':
#     import sys
#    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
seed = 42
nburn = 10000
nthin = 1
nsamples = 2000

np.random.seed(seed)

# Sample from the prior
model = init_model()

# Ground truth drawn from the model
R_true, P_true, Mu_true, Var_true = create_piecewise_constant()

Mu = R_true * P_true / (1 - P_true)

# Generate the data
Y = np.random.poisson(np.random.gamma(R_true[...,None], scale=(P_true / (1-P_true))[...,None], size=(nrows, ncols, ndepth, nreplicates)))
Y = Y.astype(float)

# Hold out some curves
Y_missing = Y.copy()
Y_missing[:3,:3] = np.nan

# Save the initial model values
X = np.arange(ndepth)

####### Run the Gibbs sampler #######
results = model.run_gibbs(Y_missing, nburn=nburn, nthin=nthin, nsamples=nsamples, print_freq=100, verbose=True)
Ws = results['W']
Vs = results['V']
Rs = results['R']
Tau2s = results['Tau2']
lam2s = results['lam2']
sigma2s = results['sigma2']

# Get the Bayes estimate
Ps = ilogit(np.einsum('znk,zmtk->znmt', Ws, Vs).clip(-10,10))
Mu_hat = Rs * Ps / (1 - Ps)
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
        ax.plot(X, Mu[i,j], color='black')
        if len(Y.shape) > 3:
            for k in range(ndepth):
                ax.scatter(np.full(Y.shape[-1],X[k]), Y[i,j,k], color='gray')
        else:
            ax.scatter(X, Y[i,j], color='gray')
        ax.plot(X, Mu_hat_mean[i,j], color='orange')
        ax.fill_between(X, Mu_hat_lower[i,j], Mu_hat_upper[i,j], color='orange', alpha=0.5)
plt.ylim([np.nanmin(Y)-0.01, np.nanmax(Y)+0.01])
plotdir = 'plots/'
if not os.path.exists(plotdir):
    os.makedirs(plotdir)
plt.savefig(os.path.join(plotdir, 'negbinom-tensor-filtering.pdf'), bbox_inches='tight')
plt.close()



