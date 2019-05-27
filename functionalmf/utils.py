import numpy as np
from scipy.sparse import issparse, coo_matrix, csc_matrix, vstack
from collections import defaultdict

def hypercube_edges(dims, use_map=False):
    '''Create edge lists for an arbitrary hypercube. TODO: this is probably not the fastest way.'''
    edges = []
    nodes = np.arange(np.product(dims)).reshape(dims)
    for i,d in enumerate(dims):
        for j in range(d-1):
            for n1, n2 in zip(np.take(nodes, [j], axis=i).flatten(), np.take(nodes,[j+1], axis=i).flatten()):
                edges.append((n1,n2))
    if use_map:
        return edge_map_from_edge_list(edges)
    return edges

def edge_map_from_edge_list(edges):
    result = defaultdict(list)
    for s,t in edges:
        result[s].append(t)
        result[t].append(s)
    return result

def matrix_from_edges(edges):
    '''Returns a sparse penalty matrix (D) from a list of edge pairs. Each edge
    can have an optional weight associated with it.'''
    max_col = 0
    cols = []
    rows = []
    vals = []
    if type(edges) is defaultdict:
        edge_list = []
        for i, neighbors in edges.items():
            for j in neighbors:
                if i <= j:
                    edge_list.append((i,j))
        edges = edge_list
    for i, edge in enumerate(edges):
        s, t = edge[0], edge[1]
        weight = 1 if len(edge) == 2 else edge[2]
        cols.append(min(s,t))
        cols.append(max(s,t))
        rows.append(i)
        rows.append(i)
        vals.append(weight)
        vals.append(-weight)
        if cols[-1] > max_col:
            max_col = cols[-1]
    return coo_matrix((vals, (rows, cols)), shape=(rows[-1]+1, max_col+1)).tocsc()

def grid_penalty_matrix(dims, k):
    edges = hypercube_edges(dims)
    D = matrix_from_edges(edges)
    return get_delta(D, k)

def get_delta(D, k):
    '''Calculate the k-th order trend filtering matrix given the oriented edge
    incidence matrix and the value of k.'''
    if k < 0:
        raise Exception('k must be at least 0th order.')
    result = D
    for i in range(k):
        result = D.T.dot(result) if i % 2 == 0 else D.dot(result)
    return result

def bayes_delta(D, K, anchor=0):
    # Start with the simple mu_1 ~ N(0, sigma)
    Dbayes = np.zeros((1,D.shape[1]))
    Dbayes[0,anchor] = 1

    if issparse(D):
        Dbayes = csc_matrix(Dbayes)

    # Add in the k'th order diffs
    for k in range(K+1):
        Dk = get_delta(D, k=k)
        if issparse(Dbayes):
            Dbayes = vstack([Dbayes, Dk])
        else:
            Dbayes = np.concatenate([Dbayes, Dk], axis=0)
    return Dbayes

def bayes_grid_penalty(dims, k, anchor=0):
    if not hasattr(dims, '__len__'):
        dims = [dims]
    if len(dims) == 1:
        D = get_1d_penalty_matrix(dims[0])
    else:
        D = grid_penalty_matrix(dims, 0)
    return bayes_delta(D, k, anchor=anchor)


def get_1d_penalty_matrix(N):
    '''Create a 1D trend filtering penalty matrix D^(k+1).'''
    rows = np.repeat(np.arange(N-1), 2)
    cols = np.repeat(np.arange(N), 2)[1:-1]
    data = np.tile([-1, 1], N-1)
    return coo_matrix((data, (rows, cols)), shape=(N-1, N)).tocsc()


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def ilogit(x):
    return 1 / (1 + np.exp(-x))

def mse(x, y):
    return np.nanmean((x - y)**2)

def mae(x, y):
    return np.nanmean(np.abs(x - y))

def sample_horseshoe_plus(size=1):
    a = 1/np.random.gamma(0.5, 1, size=size)
    b = 1/np.random.gamma(0.5, a)
    c = 1/np.random.gamma(0.5, b)
    d = 1/np.random.gamma(0.5, c)
    return d, c, b, a

def sample_horseshoe(size=1):
    a = 1/np.random.gamma(0.5, 1, size=size)
    return 1/np.random.gamma(0.5, a), a

def grid_ep_approx(likelihood, ngrid=100, x_min=0, x_max=1, tol=1e-4, min_space=1e-3):
    '''Fits an expectation propagation (AKA forward KL) model to the likelihood
    using a grid approximation. Solves the optimization problem:

    min. KL(P(x) || Q(x))

    where Q is a normal distribution and P(x) is approximated by the x grid.
    Returns a pair of parameters (Mu, Sigma) where each has shape
    grid.shape[:-1].'''

    # Create a grid of points starting with the max grid size
    grid = np.linspace(x_min, x_max, ngrid)

    # Initialize the grid probabilities
    probs = likelihood(grid)
    probs /= probs.sum()
    upper = x_max
    lower = x_min

    # Find a good grid approximation
    while probs.min() < tol:
        # Remove the smallest and largest bins
        to_remove = np.argmin(probs)
        to_split = np.argmax(probs)

        # Figure out the new bins, handling edge cases
        if grid[to_split] == x_max:
            to_add = [(grid[to_split-1]+grid[to_split])/2, grid[to_split]]
        elif grid[to_split] == x_min:
            to_add = [grid[to_split], (grid[to_split] + grid[to_split+1])/2]
        elif to_split == ngrid-1:
            to_add = np.linspace(grid[to_split-1], upper, 4)[1:3]
        elif to_split == 0:
            to_add = np.linspace(lower, grid[to_split+1])[1:3]
        else:
            to_add = np.linspace(grid[to_split-1], grid[to_split+1], 4)[1:3]

        # Handle edge cases of the smallest bin being on a boundary
        # TODO: is there a better way than just shrinking by a tiny amount?
        if to_remove == 0:
            lower = grid[to_remove] + min_space
        elif to_remove == ngrid-1:
            upper = grid[to_remove] - min_space

        # Drop the two bins
        grid = np.delete(grid, [to_remove, to_split])
        
        # Handle edge cases of removing the last bins
        if to_remove > to_split:
            to_split -= 1
        to_split = min(to_split, len(grid))

        # Add the two new bins
        grid = np.insert(grid, to_split, to_add)
        probs = likelihood(grid)
        probs /= probs.sum()

    # Find a normal distribution centered at the mean, with variance that
    # minimizes the 

    # Get the mean and standard deviation (Gaussian EP approximation)
    mu = (probs * grid).sum()
    sigma = np.sqrt((probs * (grid - mu)**2).sum())

    return mu, sigma

# CODE TO TEST EP APPROXIMATIONS
# %pylab
# from scipy.stats import norm
# mu1, sigma1 = 0.9, 0.2
# mu2, sigma2 = 1.2, 0.2
# likelihood = lambda x: 0.5*norm.pdf(x, mu1, sigma1) / (norm.cdf(1, mu1, sigma1) - norm.cdf(0, mu1, sigma1)) + 0.5*norm.pdf(x, mu2, sigma2) / (norm.cdf(1, mu2, sigma2) - norm.cdf(0, mu2, sigma2))
# mu, sigma = grid_ep_approx(likelihood)
# x = np.linspace(0,1,1000)
# plt.plot(x, likelihood(x), label='truth')
# plt.plot(x, norm.pdf(x, mu, sigma), label='EP approximation')
# plt.savefig('gmm-ep.pdf', bbox_inches='tight')
# plt.close()

# from scipy.stats import gamma
# x_min = 3
# x_max = 17
# lam1, lam2 = 5, 12
# likelihood = lambda x: 0.5*gamma.pdf(x, lam1) / (gamma.cdf(x_max, lam1) - gamma.cdf(x_min, lam1)) + 0.5*gamma.pdf(x, lam2) / (gamma.cdf(x_max, lam2) - gamma.cdf(x_min, lam2))
# mu, sigma = grid_ep_approx(likelihood, x_min=x_min, x_max=x_max)
# x = np.linspace(x_min, x_max, 1000)
# plt.plot(x, likelihood(x), label='truth')
# plt.plot(x, norm.pdf(x, mu, sigma), label='EP approximation')
# plt.savefig('gammm-ep.pdf', bbox_inches='tight')
# plt.close()


def factor_pav(W, V, in_place=False):
    '''Applies the pool adjacent violators (PAV) algorithm to the V vectors,
    ensuring the W_i . V is monotone decreasing for all i.'''
    # Reconstruct the matrix
    if not in_place:
        V = np.copy(V)
    M = W.dot(V.T)

    # check all rows for monotonicity constraint
    violators = (M[:,:-1] - M[:,1:]) < 0
    q = np.arange(V.shape[0])
    while np.any(violators):
        j = 0
        while j < V.shape[0]-1:
            # Reconstruct the current 2 columns
            M_j = W.dot(V[j:j+2].T)

            # Check for any violations
            if np.any((M_j[:,0] - M_j[:,1]) < 0):
                # Merge the two pools together by doing a weighted average
                pool0 = q == q[j]
                pool1 = q == q[j+1]
                w0 = pool0.sum()
                w1 = pool1.sum()
                V[pool0 | pool1] = (w0*V[j] + w1*V[j+1]) / (w0+w1)
                q[pool1] = q[j]
                j += w1
            else:
                j += 1

        # Check for new violators
        M = W.dot(V.T)
        violators = (M[:,:-1] - M[:,1:]) < 0

    return V

##### CODE TO TEST FACTOR_PAV #####
# import matplotlib.pyplot as plt
# import numpy as np
# nrows = 4
# ncols = 20
# nembed = 5
# W = np.random.gamma(1,1,size=(nrows, nembed))
# V = np.random.gamma(1,1,size=(ncols, nembed)).cumsum(axis=1)[::-1] + np.random.gamma(0.1,0.1,size=(ncols, nembed))

# # Project to montone curves
# V_proj = factor_pav(W, V)

# fig, axarr = plt.subplots(1, nrows, figsize=(nrows*5, 5))
# x = np.arange(ncols)
# M = W.dot(V.T)
# M_proj = W.dot(V_proj.T)
# for i in range(nrows):
#     axarr[i].scatter(x, M[i], color='gray', alpha=0.5)
#     axarr[i].plot(x, M_proj[i], color='blue')
# plt.savefig('plots/factor-pav.pdf', bbox_inches='tight')
# plt.close()

def tensor_nmf(Y, nembeds, max_steps=30, monotone=False, tol=1e-4, verbose=False, max_entry=None):
    '''Nonnegative matrix factorization for 3-tensors. Monotonicity in the
    third dimension is optionally enforced.'''
    from scipy.optimize import nnls
    from functionalmf.utils import factor_pav
    W = np.random.gamma(1,1,size=(Y.shape[0], nembeds))
    V = np.random.gamma(1,1,size=(Y.shape[1], Y.shape[2], nembeds))

    # Enforce lower triangular shape on W
    if Y.shape[0] > 1:
        W[np.triu_indices(nembeds, k=1)] = 0

    # Assume we are dealing with replicates
    if len(Y.shape) == 3:
        Y = Y[...,None]

    # Fit via alternating least squares
    rmse = np.inf
    for step in range(max_steps):
        if verbose:
            print('Step {}'.format(step))

        # Track the previous iteration to assess convergence
        prev_W, prev_V = np.copy(W), np.copy(V)
        prev_rmse = rmse

        # Fix V and fit W
        V_mat = np.repeat(V.reshape((-1, V.shape[-1])), Y.shape[-1], axis=0)
        for i in range(W.shape[0]):
            # Get the vector of observations for this row
            Y_vec = Y[i].flatten()

            # Handle missing observations
            missing = np.isnan(Y_vec)
            A = V_mat[~missing]
            b = Y_vec[~missing]

            # Enforce lower triangular shape on W
            ndims = min(W.shape[1], i+1)
            A = A[:,:ndims]

            W[i,:ndims] = nnls(A, b)[0].clip(1e-3, np.inf)

            # Project down to within the constraints
            if max_entry is not None and (W[i,None,None,:ndims] * V[...,:ndims]).sum(axis=-1).max() > max_entry:
                from scipy.optimize import minimize
                def fun(x):
                    return 0.5*((b - x.dot(A.T))**2).sum()
                cons = ({'type': 'ineq', 'fun': lambda x: max_entry - (x[None,None] * V[...,:ndims]).sum(axis=-1).flatten()},
                        {'type': 'ineq', 'fun': lambda x: (x[None,None] * V[...,:ndims]).sum(axis=-1).flatten()},
                        {'type': 'ineq', 'fun': lambda x: x - 1e-6})
                res = minimize(fun, x0=W[i,:ndims], constraints=cons, method='SLSQP', options={'ftol':1e-8, 'maxiter':1000})
                W[i,:ndims] = res.x

        # Fix W and fit V
        W_mat = np.repeat(W, Y.shape[-1], axis=0)
        for j in range(V.shape[0]):
            for k in range(V.shape[1]):
                # Get the vector of observations for this row
                Y_vec = Y[:,j,k].flatten()

                # Handle missing observations
                missing = np.isnan(Y_vec)
                A = W_mat[~missing]
                b = Y_vec[~missing]

                V[j,k] = nnls(A, b)[0].clip(1e-3, np.inf)

                # Project down to within the constraints
                if max_entry is not None and (V[None,j,k] * W).sum(axis=-1).max() > max_entry:
                    from scipy.optimize import minimize
                    def fun(x):
                        return 0.5*((b - x.dot(A.T))**2).sum()
                    cons = ({'type': 'ineq', 'fun': lambda x: max_entry - x.dot(W.T)},
                            {'type': 'ineq', 'fun': lambda x: x.dot(W.T)},
                            {'type': 'ineq', 'fun': lambda x: x - 1e-6})
                    res = minimize(fun, x0=V[j,k], constraints=cons, method='SLSQP', options={'ftol':1e-8, 'maxiter':1000})
                    V[j,k] = res.x

            # optionally project to monotone curve
            if monotone:
                factor_pav(W, V[j], in_place=True)

        # delta = np.linalg.norm(np.concatenate([(prev_W - W).flatten(), (prev_V - V).flatten()]))
        rmse = np.sqrt(np.nansum((Y - (W[:,None,None]*V[None]).sum(axis=-1,keepdims=True))**2))
        delta = (prev_rmse - rmse) / rmse

        if verbose:
            print('delta: {}'.format(delta))
        if delta <= tol:
            break

    return W, V


def ep_from_mf(Y, W, V, mode='max', multiplier=2):
    '''Over-estimates the standard deviation of errors around the matrix
    factorization prediction of the means.'''
    if len(Y.shape) == 3:
        Y = Y[...,None]
    M = (W[:,None,None] * V[None]).sum(axis=-1, keepdims=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sqerr = np.nanmean((Y - M)**2, axis=-1)
        if mode == 'max':
            overestimate = np.sqrt(np.nanmax(sqerr))
        elif mode == 'multiplier':
            overestimate = np.sqrt(np.nanmean(sqerr))*multiplier
    print('Estimated stdev: {}'.format(overestimate))
    return M[...,0], np.ones(Y.shape[:-1])*overestimate

def random_holdouts(Y, nholdout):
    print('Holding out {} random curves'.format(nholdout))
    options = [idx for idx in np.ndindex(Y.shape[:-2]) if not np.all(np.isnan(Y[idx]))]
    selected = np.array([options[i] for i in np.random.choice(len(options), replace=False, size=nholdout)])
    Y_candidate = Y.copy()
    Y_candidate[selected[:,0], selected[:,1]] = np.nan

    # Make sure the held out data points don't leave an empty column or row
    invalid = np.any(np.all(np.isnan(Y_candidate), axis=(1,2,3))) | np.any(np.all(np.isnan(Y_candidate), axis=(0,2,3)))
    while invalid:
        selected = np.array([options[i] for i in np.random.choice(len(options), replace=False, size=nholdout)])
        Y_candidate = Y.copy()
        Y_candidate[selected[:,0], selected[:,1]] = np.nan
        invalid = np.any(np.all(np.isnan(Y_candidate), axis=(1,2,3))) | np.any(np.all(np.isnan(Y_candidate), axis=(0,2,3)))
    
    # Remove the held out data points but keep track of them for evaluation at the end
    return selected




