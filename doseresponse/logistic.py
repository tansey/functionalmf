import numpy as np
from utils import load_data_as_pandas
from functionalmf.utils import ilogit, mse


def estimate_likelihood(df):
    from collections import defaultdict
    cells = list(df['cell line'].unique())
    drugs = list(df['drug'].unique())
    concentrations = [c for c in sorted(df['concentration'].unique()) if not np.isnan(c)]
    print('Concentration values:', concentrations)
    outcomes = defaultdict(list)
    controls = defaultdict(list)
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(idx)
        cell = cells.index(row['cell line'])
        drug = drugs.index(row['drug'])
        conc = row['concentration']
        outcome = row['outcome']
        if np.isnan(conc):
            controls[(cell, drug)].append(outcome)
        else:
            outcomes[(cell, drug, concentrations.index(conc))].append(outcome)

    # Estimate the gamma parameters for the controls, after shrinking to reasonable sizes
    for cell in range(len(cells)):
        for drug in range(len(drugs)):
            if (cell,drug) not in controls:
                continue
            obs = controls[(cell,drug)]
            mu, std = np.mean(obs), np.std(obs)
            for t in range(len(concentrations)):
                outcomes[(cell, drug, t)] = (np.mean(outcomes[(cell, drug, t)]) / mu).clip(0,1)

    # Build the tensor
    Y = np.full((len(cells), len(drugs), len(concentrations)), np.nan)
    for (i,j,t), o in outcomes.items():
        Y[i,j,t] = o
    return Y, cells, drugs, concentrations

def fit_logistic_factors(Y, nembeds, max_steps=100, concentrations=None, verbose=False, tol=1e-4, regularizer=1e-4):
    from scipy.optimize import minimize
    if concentrations is None:
        concentrations = np.arange(Y.shape[2])
    W = np.random.normal(0, 0.1, size=(Y.shape[0], nembeds))
    V = np.random.normal(0, 0.1, size=(Y.shape[1], nembeds))
    a, b = np.random.normal(size=(Y.shape[0])), np.random.normal(size=(Y.shape[1]))

    # Fit the embedding model via alternating minimization
    rmse = np.inf
    for step in range(max_steps):
        if verbose:
            print('Step {}'.format(step))

        # Track the previous iteration to assess convergence
        prev_W, prev_V = np.copy(W), np.copy(V)
        prev_rmse = rmse

        # Fix V and fit W
        for i in range(W.shape[0]):
            def fun(x):
                logit = np.einsum('k,mk,t->mt', x[1:], V, concentrations) + x[0] + b[:,None]
                return np.nansum((Y[i] - ilogit(logit))**2) + regularizer*(x**2).mean()
                bounds = [(-10, 10) for _ in range(nembeds+1)]
            res = minimize(fun, x0=np.concatenate([a[i:i+1], W[i]]), method='SLSQP', options={'ftol':1e-8, 'maxiter':1000}, bounds=bounds)
            a[i], W[i] = res.x[0], res.x[1:]

        # Fix W and fit V
        for j in range(V.shape[0]):
            def fun(x):
                logit = np.einsum('k,nk,t->nt', x[1:], W, concentrations) + x[0] + a[:,None]
                return np.nansum((Y[:,j] - ilogit(logit))**2) + regularizer*(x**2).mean()
            bounds = [(-10, 10) for _ in range(nembeds+1)]
            res = minimize(fun, x0=np.concatenate([b[j:j+1], V[j]]), method='SLSQP', options={'ftol':1e-8, 'maxiter':1000}, bounds=bounds)
            b[j], V[j] = res.x[0], res.x[1:]

        # delta = np.linalg.norm(np.concatenate([(prev_W - W).flatten(), (prev_V - V).flatten()]))
        Mu = ilogit(np.einsum('nk,mk,t->nmt', W, V, concentrations) + a[:,None,None] + b[None,:,None])
        rmse = np.sqrt(np.nansum((Y - Mu)**2))
        delta = (prev_rmse - rmse) / rmse

        if verbose:
            print('delta: {}'.format(delta))
        if delta <= tol:
            break

    Mu = ilogit(np.einsum('nk,mk,t->nmt', W, V, concentrations) + a[:,None,None] + b[None,:,None])

    return Mu, W, V, a, b

def select_nonempty(Y, nholdout):
    options = [idx for idx in np.ndindex(Y.shape[:2]) if not np.all(np.isnan(Y[idx]))]
    selected = np.array([options[i] for i in np.random.choice(len(options), replace=False, size=nholdout)])
    Y_candidate = Y.copy()
    Y_candidate[selected[:,0], selected[:,1]] = np.nan

    # Make sure the held out data points don't leave an empty column or row
    invalid = np.any(np.all(np.isnan(Y_candidate), axis=(1,2))) | np.any(np.all(np.isnan(Y_candidate), axis=(0,2)))
    while invalid:
        selected = np.array([options[i] for i in np.random.choice(len(options), replace=False, size=nholdout)])
        Y_candidate = Y.copy()
        Y_candidate[selected[:,0], selected[:,1]] = np.nan
        invalid = np.any(np.all(np.isnan(Y_candidate), axis=(1,2))) | np.any(np.all(np.isnan(Y_candidate), axis=(0,2)))
    return selected, Y_candidate

if __name__ == '__main__':
    import os
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    parser = argparse.ArgumentParser(description='Functional logistic matrix factorization for dose-response modeling.')
    
    # General settings
    parser.add_argument('--data', default='doseresponse/data/sim/data.csv', help='Location of the data file.')
    parser.add_argument('--outdir', default='doseresponse/data/sim/', help='Directory where all results will be saved.')
    parser.add_argument('--plot', action='store_true', help='If true, the data and results will be plotted at the end.')
    parser.add_argument('--big_plot', action='store_true', help='If true and plot is true, a single huge plot will be made.')
    
    # Model settings
    parser.add_argument('--nembeds', nargs='+', type=int, default=[1,3,5,8], help='Size of the embedding dimension.')
    parser.add_argument('--nfolds', type=int, default=5, help='The number of folds to use for CV when selecting the embedding dimension')
    parser.add_argument('--seed', type=int, default=42, help='The pseudo-random number generator seed.')
    
    # Evaluation settings
    parser.add_argument('--nholdout', type=int, default=0, help='Number of curves to hold out for test evaluation.')
    
    # Get the arguments from the command line
    args = parser.parse_args()

    # Seed the random number generator so we get reproducible results
    np.random.seed(args.seed)
    
    # Load the data
    df = load_data_as_pandas(args.data)

    print('Loading data and preprocessing')
    Y, cells, drugs, concentrations = estimate_likelihood(df)

    # Get the dimensions of the data
    nrows, ncols, ndepth = Y.shape
    X = np.arange(ndepth)
    print('Y shape: {}'.format(Y.shape))

    if args.nholdout > 0:
        print('Holding out {} random curves'.format(args.nholdout))
        # Remove the held out data points but keep track of them for evaluation at the end
        Y_full = Y.copy()
        held_out, Y = select_nonempty(Y, args.nholdout)
        print(held_out)

    print('Selecting nembeds via CV')
    folds = [((fold_idx*nrows//args.nfolds, (fold_idx+1)*nrows//args.nfolds),
              (fold_idx*ncols//args.nfolds, (fold_idx+1)*ncols//args.nfolds))
              for fold_idx in range(args.nfolds)]
    cv_results = np.zeros((args.nfolds, len(args.nembeds)))
    for fold_idx, fold in enumerate(folds):
        Y_cv = Y.copy()
        Y_cv[fold[0][0]:fold[0][1], fold[1][0]:fold[1][1]] = np.nan
        for k_idx, k in enumerate(args.nembeds):
            print('fold {} k={}'.format(fold_idx, k))
            Mu_cv, W, V, a, b = fit_logistic_factors(Y_cv, k, concentrations=concentrations, verbose=False)
            cv_results[fold_idx, k_idx] = mse(Y[fold[0][0]:fold[0][1], fold[1][0]:fold[1][1]],
                                              Mu_cv[fold[0][0]:fold[0][1], fold[1][0]:fold[1][1]])
            print(cv_results[fold_idx, k_idx])
    best_k = args.nembeds[np.argmin(cv_results.mean(axis=0))]
    print('Best K: {}'.format(best_k))

    print('Fitting the logistic factor model')
    Mu_logistic, W, V, a, b = fit_logistic_factors(Y, best_k, concentrations=concentrations, verbose=False)

    print('Saving results to file')
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    np.save(os.path.join(args.outdir, 'y_logistic'), Y)
    np.save(os.path.join(args.outdir, 'W_logistic'), W)
    np.save(os.path.join(args.outdir, 'V_logistic'), V)
    np.save(os.path.join(args.outdir, 'a_logistic'), a)
    np.save(os.path.join(args.outdir, 'b_logistic'), b)
    np.save(os.path.join(args.outdir, 'logistic_mf'), Mu_logistic)








