'''
Code to setup the data for the Google Flu Trends benchmark.

Run this before running runstuff_varinds_flu_states.m.
'''
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np
import os
from scipy.io import loadmat, savemat
from functionalmf.factor import GaussianBayesianTensorFiltering

np.random.seed(42)

df_states = loadmat('flu_US_states.mat')
Y_states = df_states['data'].T[:,None]

# Log-transform the data
Y = np.log(Y_states)
Y_train = np.log(loadmat('flu_US_states_train.mat')['data'].T[:,None])
to_hold = np.load('held_out_years.npy')


nburn = 100
nthin = 1
nsamples = 100

nrows, ncols, ndepth = Y.shape
model = GaussianBayesianTensorFiltering(nrows, ncols, ndepth, nembeds=10, tf_order=2, sigma2_init=1, nthreads=1, lam2_init=0.1, nu2_init=1)

####### Run the Gibbs sampler #######
print('Running Gibbs sampler')
results = model.run_gibbs(Y_train, nburn=nburn, nthin=nthin, nsamples=nsamples, print_freq=50, verbose=True)
Ws = results['W']
Vs = results['V']
Tau2s = results['Tau2']
lam2s = results['lam2']
sigma2s = results['sigma2']
nu2s = results['nu2']

# Look at just the first 100 instead of 1000
# Ws, Vs, Tau2s, lam2s, sigma2s, nu2s = Ws[:100], Vs[:100], Tau2s[:100], lam2s[:100], sigma2s[:100], nu2s[:100]

# Get the Bayes estimate
Mu_hat = np.einsum('znk,zmtk->znmt', Ws, Vs)
Mu_hat_mean = Mu_hat.mean(axis=0)
Mu_hat_upper = np.percentile(Mu_hat, 97.5, axis=0)
Mu_hat_lower = np.percentile(Mu_hat, 2.5, axis=0)

# Get the posterior predictives and confidence bands
# Y_samples = np.random.normal(Mu_hat, nu2s[:,None,:,None], size=[100] + [d for d in Mu_hat.shape])
# Y_samples = Y_samples.reshape([-1,Y_samples.shape[2],Y_samples.shape[3],Y_samples.shape[4]])
# Y_upper = np.percentile(Y_samples, 97.5, axis=0)
# Y_lower = np.percentile(Y_samples, 2.5, axis=0)

# If the whole thing is too big, do it iteratively
Y_lower, Y_upper = np.zeros(Y.shape), np.zeros(Y.shape)
for i in range(Y.shape[0]):
    if i % 10 == 0:
        print(i)
    for k in range(Y.shape[2]):
        Y_samples_ik = np.random.normal(Mu_hat[:,i,0,k], np.sqrt(nu2s[:,0]), size=(100,Mu_hat.shape[0]))
        Y_upper[i,0,k] = np.percentile(Y_samples_ik, 97.5)
        Y_lower[i,0,k] = np.percentile(Y_samples_ik, 2.5)

###### Plot the true curves, the noisy observations, and the fits ######
print('Plotting results')
# X = np.arange(ndepth)
X = np.array([datetime.datetime.strptime(x[0][0], '%Y-%m-%d') for x in df_states['dates']])
myears = mdates.YearLocator()   # every year
mmonths = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

plot_ncols = 10
plot_nrows = nrows // plot_ncols + int((nrows % plot_ncols) > 0)
fig, axarr = plt.subplots(plot_nrows, plot_ncols, figsize=(5*plot_ncols,5*plot_nrows), sharex=True, sharey=True)
formatter = mdates.DateFormatter("%Y-%m-%d")
for i in range(nrows):
    if i % plot_ncols == 0:
        print(i)
    ax = axarr[i // plot_ncols, i % plot_ncols]
    is_missing = np.isnan(Y[i,0])
    is_held_out = (~is_missing) & np.isnan(Y_train[i,0])
    is_in_sample = (~is_missing) & (~is_held_out)
    ax.scatter(X[~is_missing], Y[i,0,~is_missing], color='black')
    ax.plot(X, Mu_hat_mean[i,0], color='orange', lw=3)
    ax.fill_between(X, Y_lower[i,0], Y_upper[i,0], color='orange', alpha=0.5)
    for _, start, end in to_hold[to_hold[:,0] == i]:
        ax.axvspan(X[start], X[end-1], color='gray', alpha=0.3)
    ax.set_title(df_states['USnames'][i][0][0])
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
plt.ylim([np.nanmin(Y)-0.01, np.nanmax(Y)+0.01])
plotdir = 'plots/'
if not os.path.exists(plotdir):
    os.makedirs(plotdir)
plt.savefig(os.path.join(plotdir, 'gaussian-tensor-filtering-states.pdf'), bbox_inches='tight')
plt.close()

###### Check posterior predictive coverage ######
is_missing = np.isnan(Y)
is_held_out = (~is_missing) & np.isnan(Y_train)
is_in_sample = (~is_missing) & (~is_held_out)
print('In-sample  coverage: {:.2f}%'.format(100-((Y[is_in_sample] < Y_lower[is_in_sample]) | (Y[is_in_sample] > Y_upper[is_in_sample])).mean()*100))
print('Out-sample coverage: {:.2f}%'.format(100-((Y[is_held_out] < Y_lower[is_held_out]) | (Y[is_held_out] > Y_upper[is_held_out])).mean()*100))

###### Check posterior mean error ######
def rmse(mu):
    print('In-sample  RMSE: {:.2f}'.format(np.sqrt(np.mean((Y[is_in_sample] - mu[is_in_sample])**2))))
    print('Out-sample RMSE: {:.2f}'.format(np.sqrt(np.mean((Y[is_held_out] - mu[is_held_out])**2))))

def mae(mu):
    print('In-sample   MAE: {:.2f}'.format(np.mean(np.abs(Y[is_in_sample] - mu[is_in_sample]))))
    print('Out-sample  MAE: {:.2f}'.format(np.mean(np.abs(Y[is_held_out] - mu[is_held_out]))))

###### Compare to Fox and Dunson (2015) ######
bnp_mu_hat = np.loadtxt('flu-states/mu_mean.csv', delimiter=',')[:,None]


print('Fox and Dunson (2015)')
rmse(bnp_mu_hat)
print()
mae(bnp_mu_hat)
print()
print()
print('Bayesian Tensor Filtering')
rmse(Mu_hat_mean)
print()
mae(Mu_hat_mean)


####### Plot the 2D factor loadings ######
# W_mean = Ws.mean(axis=0)
W_mean = Ws.mean(axis=0)
from sklearn.decomposition import PCA
W_mean = PCA(n_components=2).fit_transform(W_mean)
fig, ax = plt.subplots()
ax.scatter(W_mean[:,0], W_mean[:,1])
for i, txt in enumerate(df_states['USnames']):
    ax.annotate(txt[0][0], (W_mean[i,0], W_mean[i,1]))
plt.savefig('plots/state-embeddings-2d.pdf', bbox_inches='tight')
plt.close()





