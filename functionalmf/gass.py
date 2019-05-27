'''Generalized analytic slice sampling (GASS) for truncated multivariate normal
priors.


Author: Wesley Tansey (co-figured out with Christopher Tosh)
Date: May 2019
'''
import matplotlib
import matplotlib.pyplot as plt
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


if __name__ == '__main__':
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




