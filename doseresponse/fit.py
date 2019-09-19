'''Bayesian tensor filtering for dose-response modeling.'''
import numpy as np
from empirical_bayes import estimate_likelihood
from utils import load_data_as_pandas
from functionalmf.factor import ConstrainedNonconjugateBayesianTensorFiltering
from functionalmf.utils import tensor_nmf, ep_from_mf, factor_pav, mse

def init_model(Y, likelihood, args):
    # Linear constraints requiring monotonicity and [0,1] means.
    # Note that we use a softened monotonicity constraint allowing a small
    # fudge factor for numerical stability.
    C_zero = np.concatenate([np.eye(ndepth), np.zeros((ndepth,1))], axis=1)
    C_mono = np.array([np.concatenate([np.zeros(i), [1,-1], np.zeros(ndepth-i-2), [-1e-2]]) for i in range(ndepth-1)])
    C_one = np.concatenate([np.eye(ndepth)*-1, np.full((ndepth,1),-1)], axis=1)
    C = np.concatenate([C_zero, C_one, C_mono], axis=0)

    # Initialize the model with a nonnegative matrix factorization on the clipped values
    W, V = tensor_nmf(Y, args.nembeds, monotone=True, max_entry=0.999, verbose=True)
    
    # Sanity check that we're starting at valid points
    Mu = (W[:,None,None] * V[None]).sum(axis=-1)
    assert Mu.min() >= 0, 'Mu range [{},{}]'.format(Mu.min(), Mu.max())
    assert Mu.max() <= 1, 'Mu range [{},{}]'.format(Mu.min(), Mu.max())

    # Get an EP approximation centered at the mean and with the variance overestimated.
    EP_approx = ep_from_mf(Y, W, V, mode='multiplier', multiplier=3)

    def rowcol_likelihood(Y_obs, WV, row=None, col=None):
        if row is not None:
            Y_obs = Y_obs[row]
        if col is not None:
            Y_obs = Y_obs[:,col]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            z = np.nansum(likelihood.logpdf(Y_obs, WV[...,None]))
        return z


    # Create the model
    model = ConstrainedNonconjugateBayesianTensorFiltering(Y.shape[0], Y.shape[1], Y.shape[2],
                                                          rowcol_likelihood,
                                                          C,
                                                          nembeds=args.nembeds,
                                                          tf_order=args.tf_order,
                                                          lam2_true=args.lam2,
                                                          ep_approx=EP_approx,
                                                          nthreads=args.nthreads)
    # Initialize at the NMF fit
    model.W, model.V = W, V

    return model

if __name__ == '__main__':
    import os
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    parser = argparse.ArgumentParser(description='Bayesian tensor filtering for dose-response modeling.')
    
    # General settings
    parser.add_argument('--data', default='doseresponse/data/sim/data.csv', help='Location of the data file.')
    parser.add_argument('--outdir', default='doseresponse/data/sim/', help='Directory where all results will be saved.')
    parser.add_argument('--plot', action='store_true', help='If true, the data and results will be plotted at the end.')
    parser.add_argument('--big_plot', action='store_true', help='If true and plot is true, a single huge plot will be made.')
    parser.add_argument('--plotdir', default='doseresponse/plots/sim/', help='Directory where all results will be saved.')

    # Model settings
    parser.add_argument('--nembeds', type=int, default=5, help='Size of the embedding dimension.')
    parser.add_argument('--tf_order', type=int, default=0, help='Smoothing order for the concentration dimension.')
    parser.add_argument('--lam2', type=float, default=1e-1, help='The global shrinkage parameter.')
    parser.add_argument('--nbins', type=int, default=20, help='Number of bins to use for the empirical Bayes likelihood estimate.')

    # MCMC settings
    parser.add_argument('--nsamples', type=int, default=5000, help='Posterior samples to keep.')
    parser.add_argument('--nburn', type=int, default=5000, help='Burn in sample size.')
    parser.add_argument('--nthin', type=int, default=1, help='Samples between saved samples.')
    parser.add_argument('--nchains', type=int, default=1, help='Number of independent Markov chains to run.')
    parser.add_argument('--seed', type=int, default=42, help='The pseudo-random number generator seed.')
    parser.add_argument('--nthreads', type=int, default=3, help='Number of threads to use for the Gibbs sampler.')

    # Evaluation settings
    parser.add_argument('--nholdout', type=int, default=0, help='Number of curves to hold out for test evaluation.')
    
    # Get the arguments from the command line
    args = parser.parse_args()

    # Seed the random number generator so we get reproducible results
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    # Load the data
    df = load_data_as_pandas(args.data)

    print('Loading data and performing empirical Bayes likelihood estimate')
    Y, likelihood, cells, drugs, concentrations, control_obs = estimate_likelihood(df, nbins=args.nbins, tensor_outcomes=True, plot=args.plot)

    np.save(os.path.join(args.outdir, 'cells'), cells)
    np.save(os.path.join(args.outdir, 'drugs'), drugs)

    present = np.any(np.any(~np.isnan(Y), axis=-1), axis=-1).sum()
    print('Shape: {} x {} x {} x {}. Total possible curves: {} Total present: {} Total missing: {}'.format(Y.shape[0], Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[0] * Y.shape[1], present, Y.shape[0] * Y.shape[1] - present))

    # Get the dimensions of the data
    nrows, ncols, ndepth, nreplicates = Y.shape
    X = np.arange(ndepth)

    # Hold out a subset of entries for validation
    if args.nholdout > 0:
        print('Holding out {} random curves'.format(args.nholdout))
        options = [idx for idx in np.ndindex(Y.shape[:-2]) if not np.all(np.isnan(Y[idx]))]
        selected = np.array([options[i] for i in np.random.choice(len(options), replace=False, size=args.nholdout)])
        Y_candidate = Y.copy()
        Y_candidate[selected[:,0], selected[:,1]] = np.nan

        # Make sure the held out data points don't leave an empty column or row
        invalid = np.any(np.all(np.isnan(Y_candidate), axis=(1,2,3))) | np.any(np.all(np.isnan(Y_candidate), axis=(0,2,3)))
        while invalid:
            selected = np.array([options[i] for i in np.random.choice(len(options), replace=False, size=args.nholdout)])
            Y_candidate = Y.copy()
            Y_candidate[selected[:,0], selected[:,1]] = np.nan
            invalid = np.any(np.all(np.isnan(Y_candidate), axis=(1,2,3))) | np.any(np.all(np.isnan(Y_candidate), axis=(0,2,3)))
        
        # Remove the held out data points but keep track of them for evaluation at the end
        held_out = selected.T
        Y_full = Y
        Y = Y_candidate
        print(held_out)

    # Get the raw NMF as a baseline
    print('Fitting NMF')
    W_nmf, V_nmf = tensor_nmf(Y, args.nembeds, max_entry=0.999, verbose=True)
    Mu_nmf = (W_nmf[:,None,None] * V_nmf[None]).sum(axis=-1)
    np.save(os.path.join(args.outdir, 'nmf_w'), W_nmf)
    np.save(os.path.join(args.outdir, 'nmf_v'), V_nmf)
    
    # Get the monotone projected NMF as a baseline
    print('Fitting Monotone NMF')
    W_nmf_proj, V_nmf_proj = tensor_nmf(Y, args.nembeds, monotone=True, max_entry=0.999)
    Mu_nmf_proj = (W_nmf_proj[:,None,None] * V_nmf_proj[None]).sum(axis=-1)


    print('Initializing model')
    model = init_model(Y, likelihood, args)
    Mu_init = (model.W[:,None,None]*model.V[None]).sum(axis=-1)

    # # Plot the initial data
    # print('Plotting initial data')
    # # fig, axarr = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows), sharex=True, sharey=True)
    # for i in range(nrows):
    #     for j in range(ncols):
    #         print(i,j)
    #         # ax = axarr[i,j]
    #         ax = plt
    #         # ax.plot(X, Mu[i,j], color='black')
    #         ax.plot(X, Mu_init[i,j], color='blue', label='NMF')
    #         if model.Mu_ep is not None:
    #             ax.errorbar(X, model.Mu_ep[i,j], yerr=model.Sigma_ep[i,j], alpha=0.5)
    #         if len(Y.shape) > 3:
    #             for k in range(ndepth):
    #                 ax.scatter(np.full(Y.shape[-1],X[k]), Y[i,j,k], color='gray')
    #         else:
    #             ax.scatter(X, Y[i,j], color='gray')
    #         plt.ylim([0, np.nanmax(Y)+0.01])
    #         plt.savefig('plots/smc/initial/{}-{}.pdf'.format(i,j), bbox_inches='tight')
    #         plt.close()

    print('Running Gibbs sampler. Settings: burn={} thin={} samples={}'.format(args.nburn, args.nthin, args.nsamples))
    results = model.run_gibbs(Y, nburn=args.nburn,
                                 nthin=args.nthin,
                                 nsamples=args.nsamples,
                                 print_freq=1)

    Ws = results['W']
    Vs = results['V']

    # Get the posterior draws of the effect parameters
    Mu_hat = np.matmul(Ws[:,None], np.transpose(Vs, [0,1,3,2])).transpose([0,2,1,3])
    Mu_hat_mean = Mu_hat.mean(axis=0)
    Mu_hat_upper = np.percentile(Mu_hat, 95, axis=0)
    Mu_hat_lower = np.percentile(Mu_hat, 5, axis=0)

    # Project the posteriors to monotone curves via PAV
    projected_embeddings = [(W_i, [factor_pav(W_i, V_ij) for V_ij in V_i]) for W_i, V_i in zip(Ws, Vs)]
    Ws_proj = np.array([W_i for W_i, V_i in projected_embeddings])
    Vs_proj = np.array([V_i for W_i, V_i in projected_embeddings])

    # Get the posterior monotone effect curves
    Mu_hat_proj = np.matmul(Ws_proj[:,None], np.transpose(Vs_proj, [0,1,3,2])).transpose([0,2,1,3])
    Mu_hat_proj_mean = Mu_hat_proj.mean(axis=0)
    Mu_hat_proj_upper = np.percentile(Mu_hat_proj, 95, axis=0)
    Mu_hat_proj_lower = np.percentile(Mu_hat_proj, 5, axis=0)

    
    print('RMSE on in-sample observations:')
    print('NMF:                     {}'.format(np.sqrt(mse(Mu_nmf[...,None], Y))))
    print('Monotone NMF:            {}'.format(np.sqrt(mse(Mu_nmf_proj[...,None], Y))))
    print('Posterior mean:          {}'.format(np.sqrt(mse(Mu_hat_mean[...,None], Y))))
    # print('Monotone posterior mean: {}'.format(np.sqrt(mse(Mu_hat_proj_mean[...,None], Y))))
    print()

    def nll(pred, data):
        return -np.nansum(likelihood.logpdf(data, pred))
    print('NLL on in-sample observations:')
    print('NMF:                     {}'.format(nll(Mu_nmf[...,None], Y)))
    print('Monotone NMF:            {}'.format(nll(Mu_nmf_proj[...,None], Y)))
    print('Posterior mean:          {}'.format(nll(Mu_hat_mean[...,None], Y)))
    # print('Monotone posterior mean: {}'.format(nll(Mu_hat_proj_mean[...,None], Y)))
    print()
    
    if args.nholdout > 0:
        print('RMSE on held out observations:')
        print('NMF:                     {}'.format(np.sqrt(mse(Mu_nmf[held_out[0], held_out[1],:,None], Y_full[held_out[0], held_out[1]]))))
        print('Monotone NMF:            {}'.format(np.sqrt(mse(Mu_nmf_proj[held_out[0], held_out[1],:,None], Y_full[held_out[0], held_out[1]]))))
        print('Posterior mean:          {}'.format(np.sqrt(mse(Mu_hat_mean[held_out[0], held_out[1],:,None], Y_full[held_out[0], held_out[1]]))))
        # print('Monotone posterior mean: {}'.format(np.sqrt(mse(Mu_hat_proj_mean[held_out[0], held_out[1],:,None], Y_full[held_out[0], held_out[1]]))))
        print()

        print('NLL on held out observations:')
        print('NMF:                     {}'.format(nll(Mu_nmf[held_out[0], held_out[1],:,None], Y_full[held_out[0], held_out[1]])))
        print('Monotone NMF:            {}'.format(nll(Mu_nmf_proj[held_out[0], held_out[1],:,None], Y_full[held_out[0], held_out[1]])))
        print('Posterior mean:          {}'.format(nll(Mu_hat_mean[held_out[0], held_out[1],:,None], Y_full[held_out[0], held_out[1]])))
        # print('Monotone posterior mean: {}'.format(nll(Mu_hat_proj_mean[held_out[0], held_out[1],:,None], Y_full[held_out[0], held_out[1]])))
        print()
    

    print('Saving results to file')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    np.save(os.path.join(args.outdir, 'y'), Y)
    np.save(os.path.join(args.outdir, 'nmf'), Mu_nmf)
    np.save(os.path.join(args.outdir, 'nmf_mono'), Mu_nmf_proj)
    np.save(os.path.join(args.outdir, 'btf'), Mu_hat)
    np.save(os.path.join(args.outdir, 'btf_w'), Ws)
    np.save(os.path.join(args.outdir, 'btf_v'), Vs)
    np.save(os.path.join(args.outdir, 'btf_mono'), Mu_hat_proj)
    np.save(os.path.join(args.outdir, 'btf_ep_sigma'), model.Sigma_ep)
    if args.nholdout > 0:
        np.save(os.path.join(args.outdir, 'held_out'), held_out)
    print()

    if args.plot:
        print('Plotting results')
        Y = Y_full # Plot the full dataset
        if not os.path.exists(args.plotdir):
            os.makedirs(args.plotdir)
        if args.big_plot:
            fig, axarr = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows), sharex=True, sharey=True)
        for i in range(nrows):
            print('Row {}/{}'.format(i+1,nrows))
            for j in range(ncols):
                if args.big_plot:
                    ax = axarr[i,j]
                else:
                    ax = plt
                ax.axhline(1, color='darkgray', alpha=0.5)
                ax.plot(X, Mu_init[i,j], color='blue', label='NMF')
                if model.Mu_ep is not None:
                    ax.errorbar(X, model.Mu_ep[i,j], yerr=model.Sigma_ep[i,j], alpha=0.5)
                if len(Y.shape) > 3:
                    for k in range(ndepth):
                        ax.scatter(np.full(Y.shape[-1],X[k]), Y[i,j,k], color='gray')
                else:
                    ax.scatter(X, Y[i,j], color='gray')
                plt.ylim([0, np.nanmax(Y)+0.01])
                ax.plot(X, Mu_hat_mean[i,j], color='orange')
                ax.fill_between(X, Mu_hat_lower[i,j], Mu_hat_upper[i,j], color='orange', alpha=0.5)
                if not args.big_plot:
                    plt.savefig(os.path.join(args.plotdir, 'sample{}-drug{}.pdf'.format(i,j)), bbox_inches='tight')
                    plt.close()
        if args.big_plot:
            plt.savefig(os.path.join(args.plotdir, 'all.pdf'), bbox_inches='tight')

    # Kill the threadpool
    model.executor.shutdown()





