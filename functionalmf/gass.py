'''Generalized analytic slice sampling (GASS) for truncated multivariate normal
priors.


Author: Wesley Tansey (co-figured out with Christopher Tosh)
Date: May 2019
'''
import numpy as np
from scipy.stats import norm
from functionalmf.fast_mvn import sample_mvn


def gass(x, Sigma, loglikelihood, Constraints,
            cur_ll=None, mu=None, verbose=False, ll_args=None,
            sparse=False, precision=False, chol_factor=False, Q_shape=None,
            ngrid=100):
    # Current log-likelihood
    if cur_ll is None:
        cur_ll = loglikelihood(x, ll_args)

    # Select slice height
    ll = cur_ll + np.log(np.random.random())

    # Sample proposal
    v = sample_mvn(Sigma, mu=np.zeros_like(x), sparse=sparse, precision=precision, chol_factor=chol_factor, Q_shape=Q_shape)

    # Mean of the gaussian
    if mu is None:
        mu = np.zeros_like(x)

    # Constraint matrix should have the inequality at the last column
    assert Constraints.shape[1] == mu.shape[0]+1

    # x must be a valid starting point
    assert np.all(Constraints[:,:-1].dot(x) >= Constraints[:,-1]), 'Invalid starting point!\n{}\nConstraints:\n{}'.format(x, (Constraints[:,:-1].dot(x) - Constraints[:,-1]).min())


    # Calculate lower bound constraints on the slice interval range
    x0 = x - mu
    a = Constraints[:,:-1].dot(x0)
    b = Constraints[:,:-1].dot(v)
    c = Constraints[:,-1] - Constraints[:,:-1].dot(mu)
    sqrt_term = a**2 + b**2 - c**2
    eps = 1e-6

    # Two cases cause the entire ellipse to be valid:
    # 1) the sqrt term is less than zero. this implies a**2 + b**2 < c**2 ==> a cos\theta + b sin\theta > d for all \theta
    # 2) a = -c. this implies the only place the constraint touches the ellipse is on the extremal point.
    # For anything else, some values of the ellipse will be invalid and must be pruned
    concerning = (sqrt_term >= 0) & (a != -c)
    if np.any(concerning):
        denom = a + c
        theta1 = 2*np.arctan((b[concerning] + np.sqrt(sqrt_term[concerning])) / denom[concerning])
        theta2 = 2*np.arctan((b[concerning] - np.sqrt(sqrt_term[concerning])) / denom[concerning])
        
        # If a^2 < c^2, we care about the complement of the region because the quadratic is convex.
        # Otherwise, the quadratic is concave and we care about the interval
        complements = a[concerning]**2 < c[concerning]**2 
        theta1_complements = theta1[complements]
        theta1_interval = theta1[~complements]
        theta2_complements = theta2[complements]
        theta2_interval = theta2[~complements]

        # Numerically approximate the intersection of the valid [-pi, pi] regions
        grid = np.linspace(-np.pi, np.pi, 10000)

        # Complements require the grid to be outside the [min,max] interval
        if np.any(complements):
            # TODO: vectorize
            for t1, t2 in zip(theta1_complements, theta2_complements):
                grid = grid[(grid <= min(t1, t2)) | (grid >= max(t1,t2))]
            
        # Intervals require the grid to be inside the [min,max] interval
        if np.any(~complements):
            theta_order = theta1_interval < theta2_interval
            theta_min = (theta_order*theta1_interval + (~theta_order)*theta2_interval).max() + eps
            theta_max = (theta_order*theta2_interval + (~theta_order)*theta1_interval).min() - eps
            grid = grid[(grid >= theta_min) & (grid <= theta_max)]
    else:
        # The entire ellipse is valid
        grid = np.linspace(-np.pi, np.pi, ngrid)

    if verbose > 1:
        np.set_printoptions(precision=3, suppress=True, linewidth=250)
        print('x:    ', x)
        print('x-mu: ', x0)
        print('v:    ', v)
        print('mu:   ', mu)
        print('')
        print('Grid points accepted:', grid)
        print('Total grid points: {}'.format(len(grid)))
        print('thetas:')
        for i, a_i, b_i, c_i, comp_i, theta1_i, theta2_i in zip(np.arange(len(concerning))[concerning],
                                                                a[concerning],
                                                                b[concerning],
                                                                c[concerning],
                                                                complements,
                                                                theta1/np.pi,
                                                                theta2/np.pi):
            print('{} a: {:.2f} b: {:.2f} c: {:.2f} complement? {} theta1: {:.2f} theta2: {:.2f}'.format(i, a_i, b_i, c_i, comp_i, theta1_i, theta2_i))

    if len(grid) == 0:
        grid_options = []
        if verbose:
            import warnings
            warnings.warn('No valid slice regions! Bug??')
    else:
        # Downsample the grid f there are more grid points than specified
        if len(grid) > ngrid:
            grid = np.random.choice(grid, size=ngrid, replace=False)

        # Quasi-Monte Carlo via grid approximation
        grid_options = x0[None]*np.cos(grid[:,None]) + v[None]*np.sin(grid[:,None]) + mu[None]
        grid_ll = loglikelihood(grid_options, ll_args) # Log-likelihood function should support batching
        grid_options = grid_options[grid_ll >= ll]
        grid_ll = grid_ll[grid_ll >= ll]

    # Uniform selection over the viable grid points
    if len(grid_options) > 0:
        selected = np.random.choice(len(grid_options))
        x = grid_options[selected]
        new_ll = grid_ll[selected]
    else:
        if verbose:
            import warnings
            warnings.warn('All theta values rejected. Possible bug or theta grid is too coarse.')
        theta = 0
        new_ll = cur_ll

    return x, new_ll


def benchmarks():
    '''Benchmarking GASS vs.
    1) naive ESS + rejection sampling
    2) logistic ESS + rejection sampling for monotonicity
    3) logistic ESS + posterior projection'''
    import matplotlib
    import matplotlib.pyplot as plt
    from functionalmf.elliptical_slice import elliptical_slice as ess
    from functionalmf.utils import ilogit, pav
    from scipy.stats import gamma
    np.random.seed(42)
    ntrials = 100
    nmethods = 5
    nobs = 3
    sample_sizes = np.array([100, 500, 1000, 5000, 10000], dtype=int)
    nsizes = len(sample_sizes)
    nburn = nsamples = sample_sizes.max()
    verbose = True


    mu_prior = np.array([0.95, 0.8, 0.75, 0.5, 0.29, 0.2, 0.17, 0.15, 0.01, 0.0001]) # monotonic curve prior
    T = len(mu_prior)
    b = 3
    min_mu, max_mu = 0.0, 1
    sigma_prior = 0.1*np.array([np.exp(-0.5*(i - np.arange(T))**2 / b) for i in range(T)]) # Squared exponential kernel
    
    # Get an empirical estimate of the logit-transformed covariance
    print('Building empirical covariance matrix for logit transformed model')
    mu_samples = np.zeros((1000, len(mu_prior)))
    for i in range(mu_samples.shape[0]):
        if i % 1 == 0:
            print('\t', i)
        mu_samples[i] = np.random.multivariate_normal(mu_prior, sigma_prior)
        while mu_samples[i].min() < min_mu or mu_samples[i].max() > max_mu or (mu_samples[i][1:] - mu_samples[i][:-1]).max() > 0:
            mu_samples[i] = np.random.multivariate_normal(mu_prior, sigma_prior)
    mu_samples_logit = np.log(mu_samples / (1-mu_samples))
    sigma_prior_logit = np.einsum('ni,nj->nij', mu_samples_logit, mu_samples_logit).mean(axis=0)
    mu_prior_logit = np.log(mu_prior / (1-mu_prior))

    mse = np.zeros((ntrials,nsizes,nmethods))
    coverage = np.zeros((ntrials, nsizes, nmethods,T), dtype=bool)
    for trial in range(ntrials):
        print('Trial {}'.format(trial))
        
        # Sample the true mean via rejection sampling
        mu_truth = np.random.multivariate_normal(mu_prior, sigma_prior)
        while mu_truth.min() < min_mu or mu_truth.max() > max_mu or (mu_truth[1:] - mu_truth[:-1]).max() > 0:
            mu_truth = np.random.multivariate_normal(mu_prior, sigma_prior)
        
        print(mu_truth)

        
        
        # Plot some data points using the true scale
        data = np.array([np.random.gamma(100, scale=mu_truth) for _ in range(nobs)]).T
        samples = np.zeros((nsamples, nmethods, T))

        xobs = np.tile(np.arange(T), (nobs, 1)).T

        # Linear constraints requiring monotonicity and [0, 1] intervals
        C_zero = np.concatenate([np.eye(T), np.zeros((T,1))+min_mu], axis=1)
        C_one = np.concatenate([np.eye(T)*-1, np.ones((T,1))*-1 ], axis=1)
        C_mono = np.array([np.concatenate([np.zeros(i), [1,-1], np.zeros(T-i-2), [0]]) for i in range(T-1)])

        # Setup the lower bound inequalities
        C_lower = np.concatenate([C_zero, C_one, C_mono], axis=0)

        # Initialize to be a simple line trending downward (i.e. reasonable guess)
        x = ((T - np.arange(T)) / T).clip(min_mu+0.01, max_mu-0.01)
        x = np.tile(x, nmethods).reshape(nmethods, T)
        
        # Convert to logits for the logistic models
        x[2] = np.log(x[2] / (1-x[2]))
        x[4] = np.log(x[4] / (1-x[4]))

        # Simple iid N(y | mu, sigma^2) likelihood
        def log_likelihood(z, ll_args):
            if len(z.shape) == len(data.shape):
                return gamma.logpdf(data[None], 100, scale=z[...,None]).sum(axis=-1).sum(axis=-1)
            return gamma.logpdf(data, 100, scale=z[...,None]).sum()

        # Log likelihood where we don't assume everything is valid
        def rejection_loglike(z, ll_args):
            if ll_args == 'logistic':
                z = ilogit(z)
            if z.min() < min_mu or z.max() > max_mu or (z[1:] - z[:-1]).max() > 0:
                return -np.inf
            return log_likelihood(z, ll_args)

        # Generalized analytic slice sampling
        cur_ll = [None]*nmethods
        for step in range(nburn+nsamples):
            if verbose and step % 1000 == 0:
                print(step)
            import warnings
            warnings.simplefilter("ignore")
            # Generalized analytic slice sampling
            x[0], cur_ll[0] = gass(x[0], sigma_prior, log_likelihood, C_lower, cur_ll=cur_ll[0], mu=mu_prior)

            # Naive ESS + rejection sampling
            x[1], cur_ll[1] = ess(x[1], sigma_prior, rejection_loglike, cur_log_like=cur_ll[1], mu=mu_prior)

            # Logistic ESS + rejection sampling
            x[2], cur_ll[2] = ess(x[2], sigma_prior_logit, rejection_loglike, cur_log_like=cur_ll[2], mu=mu_prior_logit, ll_args='logistic')

            # Naive ESS + posterior projection
            x[3], cur_ll[3] = ess(x[3], sigma_prior, log_likelihood, cur_log_like=cur_ll[3], mu=mu_prior)

            # Logistic ESS + posterior projection
            x[4], cur_ll[4] = ess(x[4], sigma_prior_logit, lambda x_prop, ll_args: log_likelihood(ilogit(x_prop), ll_args),
                                    cur_log_like=cur_ll[4], mu=mu_prior_logit)

            # Save posterior samples
            if step >= nburn:
                samples[step-nburn] = x

        # Pass the results for logistic models through the logistic function
        samples[:,(2,4)] = ilogit(samples[:,(2,4)])

        # Project the posteriors using PAV
        print('Projecting third and fourth method posteriors')
        for i in range(nsamples):
            samples[i,3] = pav(samples[i,3][::-1]).clip(0,1)[::-1]
            samples[i,4] = pav(samples[i,4][::-1]).clip(0,1)[::-1]

        for size_idx, sample_size in enumerate(sample_sizes):
            mu_hat = samples[:sample_size].mean(axis=0)
            mu_lower = np.percentile(samples[:sample_size], 5, axis=0)
            mu_upper = np.percentile(samples[:sample_size], 95, axis=0)

            np.set_printoptions(precision=2, suppress=True)
            # Calculate the mean squared error
            mse[trial,size_idx] = ((mu_truth[None] - mu_hat)**2).mean(axis=-1)

            # Calculate the 90th credible interval coverage
            coverage[trial,size_idx] = (mu_truth[None] >= mu_lower) & (mu_truth[None] <= mu_upper)
            
            print('Samples={} MSE={} Coverage={}'.format(sample_size,
                                                        mse[:trial+1,size_idx].mean(axis=0) * 1e3,
                                                        coverage[:trial+1,size_idx].mean(axis=0).mean(axis=1)))
        print()

            
        if trial == 0:
            np.save('data/gass-benchmark-samples.npy', samples)
            xobs = np.tile(np.arange(T), (nobs, 1)).T
            # plt.scatter(xobs, data)
            plt.plot(xobs[:,0], mu_hat[1], color='0.25', ls='--', label='ESS+Rejection')
            plt.plot(xobs[:,0], mu_hat[2], color='0.4', ls=':', label='ESS+Link+Rejection')
            plt.plot(xobs[:,0], mu_hat[3], color='0.65', ls='--', label='ESS+Projection')
            plt.plot(xobs[:,0], mu_hat[4], color='0.8', ls=':', label='ESS+Link+Projection')
            plt.plot(xobs[:,0], mu_hat[0], color='orange', label='GASS')
            plt.plot(xobs[:,0], mu_truth, color='black', label='Truth')
            # plt.fill_between(xobs[:,0], mu_lower, mu_upper, color='orange', alpha=0.5,)
            plt.axhline(1, color='purple', ls='--', label='Upper bound')
            plt.axhline(0, color='red', ls='--', label='Lower bound')
            plt.legend(loc='lower left', ncol=2)
            plt.savefig('plots/gass-benchmark.pdf', bbox_inches='tight')
            plt.close()

            import seaborn as sns
            methods = ['GASS', 'RS', 'LRS', 'PP', 'LPP']
            for midx, method in enumerate(methods):
                with sns.axes_style('white'):
                    plt.rc('font', weight='bold')
                    plt.rc('grid', lw=3)
                    plt.rc('lines', lw=3)
                    matplotlib.rcParams['pdf.fonttype'] = 42
                    matplotlib.rcParams['ps.fonttype'] = 42
                    # plt.scatter(xobs, data, color='gray', alpha=0.5)
                    plt.plot(xobs[:,0], mu_truth, color='black', label='Truth')
                    plt.plot(xobs[:,0], mu_hat[midx], color='orange', label=method)
                    plt.fill_between(xobs[:,0], mu_lower[midx], mu_upper[midx], color='orange', alpha=0.5)
                    plt.axhline(max_mu, color='red', ls='--', label='Upper bound')
                    plt.axhline(min_mu, color='red', ls='--', label='Lower bound')
                    plt.ylim([-0.1,1.1])
                    plt.ylabel('Gamma scale', fontsize=14)
                    plt.savefig('plots/gass-example-{}.pdf'.format(method.lower()), bbox_inches='tight')
                    plt.close()
        
            
    np.save('data/gass-benchmark-mse.npy', mse)
    np.save('data/gass-benchmark-coverage.npy', coverage)

    mse = mse * 1e3
    mse_mean, mse_stderr = mse.mean(axis=0), mse.std(axis=0) / np.sqrt(mse.shape[0])
    coverage_mean, coverage_stderr = coverage.mean(axis=-1).mean(axis=0), coverage.mean(axis=-1).std(axis=0) / np.sqrt(coverage.shape[0])
    for i in range(nmethods):
        print(' & '.join(['${:.2f} \\pm {:.2f}$'.format(m,s) for m,s in zip(mse_mean[:,i], mse_stderr[:,i])]))
    print()
    for i in range(nmethods):
        print(' & '.join(['${:.2f} \\pm {:.2f}$'.format(m,s) for m,s in zip(coverage_mean[:,i], coverage_stderr[:,i])]))

if __name__ == '__main__':
    benchmarks()
    np.random.seed(42)
    nobs = 5
    nburn, nsamples = 10000, 10000
    verbose = True

    # Truth
    mu_truth = np.array([1, 0.8, 0.75, 0.3, 0.29, 0.28, 0.275, 0.1, 0.01, -0.3])
    T = len(mu_truth)
    sigma_truth = 0.3
    data = np.array([np.random.normal(mu_truth, sigma_truth) for _ in range(nobs)]).T

    xobs = np.tile(np.arange(T), (nobs, 1)).T
    plt.scatter(xobs, data)
    plt.plot(xobs[:,0], mu_truth, color='black', label='Truth')
    plt.legend(loc='upper right')
    plt.savefig('plots/gass.pdf', bbox_inches='tight')
    plt.close()

    # Linear constraints requiring monotonicity and [0, 1] intervals
    C_zero = np.concatenate([np.eye(T), np.zeros((T,1))+0.1], axis=1)
    C_one = np.concatenate([np.eye(T), np.ones((T,1))], axis=1)
    C_mono = np.array([np.concatenate([np.zeros(i), [1,-1], np.zeros(T-i-2), [0]]) for i in range(T-1)])

    # Setup the lower bound inequalities
    C_lower = np.concatenate([C_zero, C_one*-1, C_mono], axis=0)

    # Priors
    mu_prior = np.full(T, 0.5)
    sigma_prior = 1

    # Initialize via a draw from the truncated prior
    x = ((T - np.arange(T)) / T).clip(0.15, 0.99)
    samples = np.zeros((nsamples, T))

    # Simple iid N(y | mu, sigma^2) likelihood
    def log_likelihood(z, ll_args):
        if len(z.shape) == len(data.shape):
            return norm.logpdf(data[None], z[...,None], sigma_truth).sum(axis=-1).sum(axis=-1)
        return norm.logpdf(data, z[...,None], sigma_truth).sum()

    # Generalized analytic slice sampling loop
    cur_ll = None
    for step in range(nburn+nsamples):
        if verbose and step % 100 == 0:
            print(step)

        # Generalized analytic slice sampling
        x, cur_ll = gass(x, sigma_prior, log_likelihood, C_lower, cur_ll=cur_ll, mu=mu_prior, verbose=True)

        # Save posterior samples
        if step >= nburn:
            samples[step-nburn] = x

    mu_hat = samples.mean(axis=0)
    mu_lower = np.percentile(samples, 5, axis=0)
    mu_upper = np.percentile(samples, 95, axis=0)

    xobs = np.tile(np.arange(T), (nobs, 1)).T
    plt.scatter(xobs, data)
    plt.plot(xobs[:,0], mu_truth, color='black', label='Truth')
    plt.plot(xobs[:,0], mu_hat, color='orange', label='Bayes estimate')
    plt.fill_between(xobs[:,0], mu_lower, mu_upper, color='orange', alpha=0.5,)
    plt.axhline(1, color='purple', ls='--', label='Upper bound')
    plt.axhline(0.1, color='red', ls='--', label='Lower bound')
    plt.legend(loc='lower left')
    plt.savefig('plots/gass.pdf', bbox_inches='tight')
    plt.close()

