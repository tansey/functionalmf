'''
Code to setup the data for the Politics benchmark.

Uses data from the GDELT project, filtered down to the G20 nations (technically just 19 nations)
'''
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np
import os
from scipy.stats import poisson
from functionalmf.factor import NonconjugateBayesianTensorFiltering, \
                                ConstrainedNonconjugateBayesianTensorFiltering, \
                                GaussianBayesianTensorFiltering,\
                                NegativeBinomialBayesianTensorFiltering
from functionalmf.pgds import fit_pgds
from functionalmf.utils import tensor_nmf, ilogit


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

def ess_loglikelihood(W, V, Y):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # mu = np.einsum('nk,mzk->nmz', W, V).clip(-6, np.inf)
        # mu[mu <= 5] = np.log1p(np.exp(mu[mu <= 5])).clip(1e-6, np.inf)
        mu = np.exp(np.einsum('nk,mzk->nmz', W, V).clip(-1e6, 1e6))
        z = np.nansum(poisson.logpmf(Y, mu))
        if ~np.all(np.isfinite(mu)):
            print('Went to inf or nan:')
            print(mu.max(), mu.min(), np.isnan(mu).sum())
            return -np.inf
    return z    

def ep_from_nmf(Y, W, V):
    if len(Y.shape) == 3:
        Y = Y[...,None]
    M = (W[:,None,None] * V[None]).sum(axis=-1, keepdims=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        estimate = np.nanmedian(np.nanmean((Y - M)**2 / M**2, axis=-1))
    print('Estimated stdev: {}'.format(estimate))
    return M[...,0], np.ones(Y.shape[:-1])*estimate


np.random.seed(42)

Y = np.load('cooperate.npy')
Y_train = np.load('cooperate_train.npy')
to_hold = np.load('held_out.npy')

nburn = 10000
nthin = 10
nsamples = 1000
nembeds = 5
nrows, ncols, ndepth = Y.shape


####### Compare to PGDS #######
Mu_pgds, (W_pgds, V_pgds, U_pgds) = fit_pgds(Y_train, nembeds, # Number of embeddings to use in the decomposition
                                                binary = False, # Set this to True if your data is binary. Otherwise, False for count data.
                                                nthreads = 1, # None will grab the max number of available threads for parallelization.
                                                time_mode=2, # Dimension of the tensor that should be smoothed in time
                                                nburn=nburn,
                                                nthin=nthin,
                                                nsamples=nsamples,
                                                verbose = 0)


# Constraints requiring positive means
C_zero = np.concatenate([np.eye(ndepth), np.zeros((ndepth,1))], axis=1)

# Setup the lower bound inequalities
model = ConstrainedNonconjugateBayesianTensorFiltering(nrows, ncols, ndepth,
                                                      rowcol_loglikelihood,
                                                      C_zero,
                                                      nembeds=nembeds, tf_order=2,
                                                      sigma2_init=0.5, nthreads=3,
                                                      lam2_init=0.1)

# Use NMF to initialize the model
model.W, model.V = tensor_nmf(Mu_pgds.mean(axis=0), nembeds)
model.Mu_ep, model.Sigma_ep = ep_from_nmf(Y_train, model.W, model.V)

'''
# Setup the non-conjugate model
model = NonconjugateBayesianTensorFiltering(nrows, ncols, ndepth, ess_loglikelihood, nembeds=nembeds, tf_order=2, sigma2_init=1, lam2_init=0.1)
# model.W, model.V = tensor_nmf(Y_train, nembeds)
'''
print('Running Gibbs sampler')
results = model.run_gibbs(Y_train, nburn=nburn, nthin=nthin, nsamples=nsamples, print_freq=10, verbose=True)
Ws = results['W']
Vs = results['V']
Tau2s = results['Tau2']
lam2s = results['lam2']
sigma2s = results['sigma2']

# Get the Bayes estimate
Mu_hat = np.einsum('znk,zmtk->znmt', Ws, Vs)
Mu_hat_mean = Mu_hat.mean(axis=0)
Mu_hat_upper = np.percentile(Mu_hat, 97.5, axis=0)
Mu_hat_lower = np.percentile(Mu_hat, 2.5, axis=0)

'''
####### Compare to Gaussian BTF ########
GTF_model = GaussianBayesianTensorFiltering(nrows, ncols, ndepth, nembeds=nembeds, tf_order=2, sigma2_init=1, nthreads=1, lam2_init=0.1, nu2_init=1)

print('Running Gibbs sampler')
GTF_results = GTF_model.run_gibbs(np.sqrt(Y_train), nburn=nburn, nthin=nthin, nsamples=nsamples, print_freq=100, verbose=True)
GTF_Ws = GTF_results['W']
GTF_Vs = GTF_results['V']
GTF_Tau2s = GTF_results['Tau2']
GTF_lam2s = GTF_results['lam2']
GTF_sigma2s = GTF_results['sigma2']
GTF_nu2s = GTF_results['nu2']

# Get the Bayes estimate
GTF_Mu_hat = np.einsum('znk,zmtk->znmt', GTF_Ws, GTF_Vs)**2
GTF_Mu_hat_mean = GTF_Mu_hat.mean(axis=0)
GTF_Mu_hat_upper = np.percentile(GTF_Mu_hat, 97.5, axis=0)
GTF_Mu_hat_lower = np.percentile(GTF_Mu_hat, 2.5, axis=0)
'''
'''
####### Compare to Negative Binomial BTF ########
model = NegativeBinomialBayesianTensorFiltering(nrows, ncols, ndepth,
                                              nembeds=nembeds, tf_order=2,
                                              sigma2_init=0.5, nthreads=1,
                                              lam2_init=0.1, nu2_init=1,
                                              rdims=(0,1,2))

results = model.run_gibbs(Y_train, nburn=nburn, nthin=nthin, nsamples=nsamples, print_freq=1000, verbose=True)
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
'''



###### Benchmark ######
is_missing = np.isnan(Y)
is_held_out = (~is_missing) & np.isnan(Y_train)
is_in_sample = (~is_missing) & (~is_held_out)

def rmse(mu):
    print('In-sample  RMSE: {:.2f}'.format(np.sqrt(np.mean((Y[None,is_in_sample] - mu[:,is_in_sample])**2, axis=-1)).mean()))
    print('Out-sample RMSE: {:.2f}'.format(np.sqrt(np.mean((Y[None,is_held_out] - mu[:,is_held_out])**2, axis=-1)).mean()))

def mae(mu):
    print('In-sample   MAE: {:.2f}'.format(np.mean(np.abs(Y[None,is_in_sample] - mu[:,is_in_sample]), axis=-1).mean()))
    print('Out-sample  MAE: {:.2f}'.format(np.mean(np.abs(Y[None,is_held_out] - mu[:,is_held_out]), axis=-1).mean()))


def log_likelihood(mu):
    print('In-sample    LL: {:.2f}'.format(poisson.logpmf(Y[None,is_in_sample], mu[:,is_in_sample]).mean(axis=-1).mean()))
    print('Out-sample   LL: {:.2f}'.format(poisson.logpmf(Y[None,is_held_out], mu[:,is_held_out]).mean(axis=-1).mean()))

print('Empirical mean')
Mu_emp = (np.ones_like(Y_train)*np.nanmean(Y_train, axis=-1)[...,None])[None]
rmse(Mu_emp)
print()
mae(Mu_emp)
print()
log_likelihood(Mu_emp)
print()
print()
print('Schein et al (2016)')
rmse(Mu_pgds)
print()
mae(Mu_pgds)
print()
log_likelihood(Mu_pgds)
print()
print()
print('BTF')
rmse(Mu_hat)
print()
mae(Mu_hat)
print()
log_likelihood(Mu_hat)



###### Plot the true curves, the noisy observations, and the fits ######
print('Plotting results')
# X = np.arange(ndepth)
X = np.array([datetime.datetime.strptime(x, '%Y-%m-%d') for x in np.load('dates.npy')])
nations = np.load('nations.npy')
myears = mdates.YearLocator()   # every year
mmonths = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

plot_ncols = Y.shape[1]
plot_nrows = Y.shape[0]
fig, axarr = plt.subplots(plot_nrows, plot_ncols, figsize=(5*plot_ncols,5*plot_nrows), sharex=True, sharey=True)
formatter = mdates.DateFormatter("%Y-%m-%d")
for i in range(nrows):
    print(i)
    for j in range(ncols):
        ax = axarr[i,j]
        is_missing = np.isnan(Y[i,j])
        is_held_out = (~is_missing) & np.isnan(Y_train[i,j])
        is_in_sample = (~is_missing) & (~is_held_out)
        ax.scatter(X[~is_missing], np.sqrt(Y[i,j,~is_missing]), color='black')
        ax.plot(X, np.sqrt(Mu_hat_mean[i,j]), color='orange', lw=3)
        ax.fill_between(X, np.sqrt(Mu_hat_lower[i,j]), np.sqrt(Mu_hat_upper[i,j]), color='orange', alpha=0.5)
        # ax.plot(X, np.sqrt(Mu_pgds[:,i,j].mean(axis=0)), color='purple', lw=3, alpha=0.5)
        if len(to_hold.shape) == 4:
            # Held out specific time periods
            for start, end in to_hold[(to_hold[:,0] == i) & (to_hold[:,1] == j)][:,2:]:
                ax.axvspan(X[start], X[end-1], color='gray', alpha=0.3)
        elif np.any((to_hold[:,0] == i) & (to_hold[:,1] == j)):
            # Held out entire (i,j) blocks
            ax.axvspan(X[0], X[-1], color='gray', alpha=0.3)
        if i == 0:
            ax.set_title(nations[j])
        if j == 0:
            ax.set_ylabel(nations[i], rotation=90)
        ax.xaxis.set_major_locator(myears)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(mmonths)
        # format the coords message box
        ax.format_xdata = formatter
        ax.grid(True)
        ax.tick_params(axis='both', which='both', labelsize=9, labelbottom=True)
        for tick in ax.get_xticklabels():
            tick.set_visible(True)

# round to nearest years.
datemin = np.datetime64(X[0], 'Y')
datemax = np.datetime64(X[-1], 'Y') + np.timedelta64(1, 'Y')
plt.xlim(datemin, datemax)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()
plt.ylim([np.nanmin(np.sqrt(Y))-0.01, np.nanmax(np.sqrt(Y))+0.01])
plotdir = 'plots/'
if not os.path.exists(plotdir):
    os.makedirs(plotdir)
plt.savefig(os.path.join(plotdir, 'poisson-tensor-filtering-politics.pdf'), bbox_inches='tight')
plt.close()




