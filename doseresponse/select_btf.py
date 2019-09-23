import numpy as np
import os
import argparse
import pandas as pd
from empirical_bayes import estimate_likelihood
from utils import load_data_as_pandas


def mu_loglikelihood(Y, Mu, likelihood):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        z = np.nansum(likelihood.logpdf(Y, Mu[...,None]))
    return z

def dic(Y, Mu, likelihood):
    # Calculate the DIC score:
    # dev(beta_i) = -2 log(p(y | beta_i))
    # DIC = 2 * avg(dev(beta)) - dev(avg(beta))
    Mu_mean = Mu.mean(axis=0)
    D_mean = -2 * mu_loglikelihood(Y, Mu_mean, likelihood)
    mean_D =  -2 * np.mean([mu_loglikelihood(Y, M, likelihood) for M in Mu])
    return 2*mean_D - D_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select hyperparameters for BTF using DIC.')
    
    # General settings
    parser.add_argument('--data', default='data/cumc.csv', help='Location of the data file.')
    parser.add_argument('--basedir', default='doseresponse/data/', help='Directory where all results will be saved.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1,2,3,4,5], help='Different random seeds used for independent trials.')
    parser.add_argument('--nembeds', nargs='+', type=int, default=[3,5,8,10,15], help='Different embedding sizes in the grid search.')
    parser.add_argument('--tf_order', nargs='+', type=int, default=[0,1], help='Different tf_order values in the grid search.')
    parser.add_argument('--lam2', nargs='+', type=int, default=[1e-3,1e-2,1e-1], help='Different lam2 values used in the grid search.')
    parser.add_argument('--nbins', type=int, default=20, help='Number of bins to use for the empirical Bayes likelihood estimate.')
    parser.add_argument('--nthin', type=int, default=1, help='Thinning for the posterior sample set.')

    # Get the arguments from the command line
    args = parser.parse_args()

    # Load the data
    df = load_data_as_pandas(args.data)

    print('Loading data and performing empirical Bayes likelihood estimate')
    Y_full, likelihood, cells, drugs, concentrations, control_obs = estimate_likelihood(df, nbins=args.nbins, tensor_outcomes=True)

    results = np.full((len(args.seeds), len(args.nembeds), len(args.tf_order), len(args.lam2)), np.nan)
    results_mono = np.copy(results)
    for sidx, seed in enumerate(args.seeds):
        print('Loading seed: {}'.format(seed))
        for kidx, emb in enumerate(args.nembeds):
            for tidx, tf in enumerate(args.tf_order):
                for lidx, lam in enumerate(args.lam2):
                    fmt = 'k{}_t{}_l{}_s{}'
                    curdir = os.path.join(args.basedir, fmt.format(emb, tf, lam, seed))

                    Y_train = np.load(os.path.join(curdir, 'y.npy'))
                    Mu_hat = np.load(os.path.join(curdir, 'btf.npy'))
                    # Mu_hat_proj = np.load(os.path.join(curdir, 'btf_mono.npy'))

                    if args.nthin > 1:
                        Mu_hat = Mu_hat[::args.nthin]
                        # Mu_hat_proj = Mu_hat_proj[::args.nthin]

                    results[sidx, kidx, tidx, lidx] = dic(Y_train, Mu_hat, likelihood)
                    # results_mono[sidx, kidx, tidx, lidx] = dic(Y_train, Mu_hat_proj, likelihood)

                    print(seed, emb, tf, lam)
                    print(results[sidx, kidx, tidx, lidx])
                    print(results_mono[sidx, kidx, tidx, lidx])
                    
    with open(os.path.join(args.basedir, 'selection_results.txt'), 'w') as f:
        for sidx, seed in enumerate(args.seeds):
            sel_k, sel_t, sel_l = np.unravel_index(results[sidx].argmin(), results.shape[1:])
            print('Raw  seed: {} nembeds: {} tf_order: {} lam2: {}'.format(seed, args.nembeds[sel_k], args.tf_order[sel_t], args.lam2[sel_l]), file=f)

            sel_k, sel_t, sel_l = np.unravel_index(results_mono[sidx].argmin(), results_mono.shape[1:])
            print('Proj seed: {} nembeds: {} tf_order: {} lam2: {}'.format(seed, args.nembeds[sel_k], args.tf_order[sel_t], args.lam2[sel_l]), file=f)
            print('',file=f)

                    








