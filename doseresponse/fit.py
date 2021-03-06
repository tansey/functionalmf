'''Bayesian tensor filtering for dose-response modeling.'''
import numpy as np
from empirical_bayes import estimate_likelihood
from utils import load_data_as_pandas
from functionalmf.factor import ConstrainedNonconjugateBayesianTensorFiltering
from functionalmf.utils import tensor_nmf, ep_from_mf, factor_pav, mse, mae
import SharedArray as sa

sharedprefix = None
worker_model = None
worker_Y = None
worker_X = None
worker_U = None
def worker_init(model):
    global worker_model
    global worker_Y
    global worker_X
    global worker_U
    worker_model = model
    worker_Y = sa.attach(model.sharedprefix + 'Y_obs')
    try:
        worker_X = sa.attach(model.sharedprefix + 'X')
        worker_U = sa.attach(model.sharedprefix + 'U')
    except:
        worker_X = None
        worker_U = None

def rowcol_likelihood(_, WV, W, V, row=None, col=None):
    Y_obs = worker_Y
    if row is not None:
        Y_obs = Y_obs[row]
    if col is not None:
        Y_obs = Y_obs[:,col]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        z = np.nansum(likelihood.logpdf(Y_obs, WV[...,None]))
    return z

def rowcol_likelihood_with_X(_, WV, W, V, row=None, col=None):
    z = rowcol_likelihood(_, WV, W, V, row=row, col=col)
    if row is not None:
        X = worker_X
        U = worker_U
        WU = W.dot(U[:,:W.shape[-1]].T)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            z += np.nansum(X[row]*np.log(WU) + (1-X[row])*np.log(1-WU), axis=-1)
    return z



def init_model(Y, likelihood, args):
    # Linear constraints requiring monotonicity and [0,1] means.
    # Note that we use a softened monotonicity constraint allowing a small
    # fudge factor for numerical stability.
    C_zero = np.concatenate([np.eye(ndepth), np.zeros((ndepth,1))], axis=1)
    C_mono = np.array([np.concatenate([np.zeros(i), [1,-1], np.zeros(ndepth-i-2), [-1e-2]]) for i in range(ndepth-1)])
    C_one = np.concatenate([np.eye(ndepth)*-1, np.full((ndepth,1),-1)], axis=1)
    C = np.concatenate([C_zero, C_one, C_mono], axis=0)

    # If the user provided an optional set of binary row features
    if args.features is not None:
        import pandas as pd

        print('Loading features')
        df = pd.read_csv(args.features, index_col=0, header=0)

        # Filter the features into those with and without dose-response data
        cells = np.load(os.path.join(args.outdir, 'cells.npy'))
        
        # Print some info on the breakdown of features and dose-response data
        have_both = [c for c in cells if c in df.index]
        doseresponse_only = [c for c in cells if c not in df.index]
        features_only = [c for c in df.index if c not in cells]
        print('Have dose-response and features: {}'.format(len(have_both)))
        print('Dose-response only: {}'.format(len(doseresponse_only)))
        print('Features only: {}'.format(len(features_only)))

        # Create feature matrices for samples with and without dose-response curves
        X_with = np.array([df.loc[c].values if c in df.index else np.full(len(df.columns), np.nan) for c in cells])
        X_without = np.array([df.loc[c].values for c in features_only])

        print('Initializing dose-response embeddings via NMF with row features')
        W, V, U = tensor_nmf(Y, args.nembeds, monotone=True, max_entry=0.999, verbose=False, row_features=X_with)

        # If we have samples that have no dose-response, generate factors for them as well
        # TODO: fitting this jointly is probably marginally better, but let's not do it for now.
        # if X_without.shape[0] > 0:
        #     W_without, _ = tensor_nmf(X_without[:,:,None], args.nembeds, V=U, fit_V=False, max_entry=0.999, verbose=False)
        X = X_with # Quick and dirty approach that just uses the samples with dose-response for now

        # Create shared arrays
        X_shared = sa.create(args.sharedprefix + 'X', X.shape)
        X_shared[:] = X
        X = X_shared
        U_shared = sa.create(args.sharedprefix + 'U', U.shape)
        U_shared[:] = U
        U = U_shared
        
        if args.sample_features:
            # Create constraints for WU to be in [0,1]
            Row_zero = np.concatenate([U,np.full((U.shape[0],1), 0)], axis=1)
            Row_one = np.concatenate([U*-1,np.full((U.shape[0],1), -1)], axis=1)
            Row_constraints = np.concatenate([Row_zero, Row_one], axis=0)

            # Posterior samples
            # U_samples = sa.create(args.sharedprefix + 'U_samples', (args.nsamples, U.shape[0], U.shape[1]))
            U_samples = np.zeros((args.nsamples, U.shape[0], U.shape[1]))

            from functionalmf.gass import gass
            def U_step(model, _, step):
                # TODO: Parallelize this
                # Setup the [0,1] constraints
                U_zero = np.concatenate([model.W,np.full((model.W.shape[0],1), 0)], axis=1)
                U_one = np.concatenate([model.W*-1,np.full((model.W.shape[0],1), -1)], axis=1)
                U_constraints = np.concatenate([U_zero, U_one], axis=0)

                U_Sigma = np.eye(U.shape[1])

                # Sample each U_i vector
                for i in range(U.shape[0]):
                    def u_loglike(u, xx):
                        if len(u.shape) == 2:
                            wu = u.dot(model.W.T)
                            return np.nansum(X[None,:,i]*np.log(wu) + (1-X[None,:,i])*np.log(1-wu), axis=1)
                        wu = model.W.dot(u)
                        return np.nansum(X[:,i]*np.log(wu) + (1-X[:,i])*np.log(1-wu))
                    U[i], _ = gass(U[i], U_Sigma, u_loglike, U_constraints)
                    

                # Update W constraints for WU to be in [0,1]
                Row_zero = np.concatenate([U,np.full((U.shape[0],1), 0)], axis=1)
                Row_one = np.concatenate([U*-1,np.full((U.shape[0],1), -1)], axis=1)
                Row_constraints = np.concatenate([Row_zero, Row_one], axis=0)
                model.Row_constraints[:] = Row_constraints

                # Save the U sample
                if step >= args.nburn and (step-args.nburn) % args.nthin == 0:
                    sidx = (step - args.nburn) // args.nthin
                    U_samples[sidx] = U

            callback = U_step
            loglikelihood = rowcol_likelihood_with_X
        else:
            Row_constraints = None
            callback = None
            U_samples = U[None]
            loglikelihood = rowcol_likelihood_with_X
    else:
        # Initialize the model with a nonnegative matrix factorization on the clipped values
        print('Initializing dose-response embeddings via NMF')
        W, V = tensor_nmf(Y, args.nembeds, monotone=True, max_entry=0.999, verbose=False)
        Row_constraints = None
        callback = None
        U_samples = None
        loglikelihood = rowcol_likelihood
    
    # Sanity check that we're starting at valid points
    Mu = (W[:,None,None] * V[None]).sum(axis=-1)
    assert Mu.min() >= 0, 'Mu range [{},{}]'.format(Mu.min(), Mu.max())
    assert Mu.max() <= 1, 'Mu range [{},{}]'.format(Mu.min(), Mu.max())

    # Get an EP approximation centered at the mean and with the variance overestimated.
    EP_approx = ep_from_mf(Y, W, V, mode='multiplier', multiplier=3)


    # Create the model
    model = ConstrainedNonconjugateBayesianTensorFiltering(Y.shape[0], Y.shape[1], Y.shape[2],
                                                          loglikelihood,
                                                          C,
                                                          nembeds=args.nembeds,
                                                          tf_order=args.tf_order,
                                                          lam2_true=args.lam2,
                                                          ep_approx=EP_approx,
                                                          nthreads=args.nthreads,
                                                          W_true=W if args.features is not None and not args.sample_features else None, # Do not sample W if we have features
                                                          Row_constraints=Row_constraints, # Row feature constraints to [0,1]
                                                          sharedprefix=args.sharedprefix,
                                                          worker_init=worker_init
                                                          )

    # Initialize at the NMF fit
    model.W[:], model.V[:] = W, V

    return model, U_samples, callback

def try_delete(name):
    try:
        sa.delete(name)
    except:
        pass

def clean_up(sharedprefix):
    # In case we exited early, clean up stuff -- TOOD: make this automatic in BTF
    try_delete(args.sharedprefix + 'X')
    try_delete(args.sharedprefix + 'U')
    try_delete(args.sharedprefix + 'Y_obs')
    try_delete(args.sharedprefix + 'W')
    try_delete(args.sharedprefix + 'V')
    try_delete(args.sharedprefix + 'sigma2')
    try_delete(args.sharedprefix + 'lam2')
    try_delete(args.sharedprefix + 'Tau2')
    try_delete(args.sharedprefix + 'Constraints_A')
    try_delete(args.sharedprefix + 'Constraints_C')
    try_delete(args.sharedprefix + 'Row_constraints')
    try_delete(args.sharedprefix + 'Mu_ep')
    try_delete(args.sharedprefix + 'Sigma_ep')
    try_delete(args.sharedprefix + 'Delta_data')
    try_delete(args.sharedprefix + 'Delta_row')
    try_delete(args.sharedprefix + 'Delta_col')

if __name__ == '__main__':
    import os
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser(description='Bayesian tensor filtering for dose-response modeling.')
    
    # General settings
    parser.add_argument('--data', default='doseresponse/data/sim/data.csv', help='Location of the data file.')
    parser.add_argument('--outdir', default='doseresponse/data/sim/', help='Directory where all results will be saved.')
    parser.add_argument('--plot', action='store_true', help='If true, the data and results will be plotted at the end.')
    parser.add_argument('--big_plot', action='store_true', help='If true and plot is true, a single huge plot will be made.')
    parser.add_argument('--plotdir', default='doseresponse/plots/sim/', help='Directory where all results will be saved.')

    # Model settings
    parser.add_argument('--nembeds', type=int, default=5, help='Size of the embedding dimension.')
    parser.add_argument('--tf_order', type=int, default=2, help='Smoothing order for the concentration dimension.')
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

    # Side-information (row features)
    parser.add_argument('--features', help='An optional matrix of binary features for each row.')
    parser.add_argument('--sample_features', action='store_true', help='If specified, samples feature embeddings jointly with dose-response embeddings; otherwise, features are fixed at monotone NMF embedding values.')
    parser.add_argument('--sharedprefix', default='dose', help='The shared memory prefix to prepend to all shared memory object names.')

    # Get the arguments from the command line
    args = parser.parse_args()

    # Seed the random number generator so we get reproducible results
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    # This (hopefully) takes care of an issue with multiprocessing running slow on AWS
    # https://stackoverflow.com/questions/59374939/aws-ec2-python-parallel-extremely-slow
    os.environ["OMP_NUM_THREADS"] = "1"

    clean_up(args.sharedprefix)

    # Load the data
    df = load_data_as_pandas(args.data)

    print('Loading data and performing empirical Bayes likelihood estimate')
    Y, likelihood, cells, drugs, concentrations, control_obs = estimate_likelihood(df, nbins=args.nbins, tensor_outcomes=True, plot=args.plot)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    np.save(os.path.join(args.outdir, 'cells'), cells)
    np.save(os.path.join(args.outdir, 'drugs'), drugs)

    present = np.any(np.any(~np.isnan(Y), axis=-1), axis=-1).sum()
    print('Shape: {} x {} x {} x {}. Total possible curves: {} Total present: {} Total missing: {}'.format(Y.shape[0], Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[0] * Y.shape[1], present, Y.shape[0] * Y.shape[1] - present))

    # Get the dimensions of the data
    nrows, ncols, ndepth, nreplicates = Y.shape
    X = np.arange(ndepth)

    # Hold out a subset of entries for validation
    Y_full = Y
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
        Y = Y_candidate
        print(held_out)

    # Create the Y in shared memory for parallel processing
    Y_shared = sa.create(args.sharedprefix + 'Y_obs', Y.shape)
    Y_shared[:] = Y
    Y = Y_shared

    # Get the raw NMF as a baseline
    print('Fitting NMF')
    W_nmf, V_nmf = tensor_nmf(Y, args.nembeds, max_entry=0.999, verbose=False)
    Mu_nmf = (W_nmf[:,None,None] * V_nmf[None]).sum(axis=-1)
    np.save(os.path.join(args.outdir, 'nmf_w'), W_nmf)
    np.save(os.path.join(args.outdir, 'nmf_v'), V_nmf)
    
    # Get the monotone projected NMF as a baseline
    print('Fitting Monotone NMF')
    W_nmf_proj, V_nmf_proj = tensor_nmf(Y, args.nembeds, monotone=True, max_entry=0.999)
    Mu_nmf_proj = (W_nmf_proj[:,None,None] * V_nmf_proj[None]).sum(axis=-1)


    print('Initializing model')
    model, Us, callback = init_model(Y, likelihood, args)
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
    results = model.run_gibbs(np.random.random(size=(3,5,2,2,2)), nburn=args.nburn,
                                 nthin=args.nthin,
                                 nsamples=args.nsamples,
                                 print_freq=1,
                                 callback=callback)

    Ws = results['W']
    Vs = results['V']

    # Get the posterior draws of the effect parameters
    Mu_hat = np.matmul(Ws[:,None], np.transpose(Vs, [0,1,3,2])).transpose([0,2,1,3])
    Mu_hat_mean = Mu_hat.mean(axis=0)
    Mu_hat_upper = np.percentile(Mu_hat, 95, axis=0)
    Mu_hat_lower = np.percentile(Mu_hat, 5, axis=0)

    # TODO -- calculate posterior predictive intervals

    # Project the posteriors to monotone curves via PAV
    projected_embeddings = [(W_i, [factor_pav(W_i, V_ij) for V_ij in V_i]) for W_i, V_i in zip(Ws, Vs)]
    Ws_proj = np.array([W_i for W_i, V_i in projected_embeddings])
    Vs_proj = np.array([V_i for W_i, V_i in projected_embeddings])

    # Get the posterior monotone effect curves
    Mu_hat_proj = np.matmul(Ws_proj[:,None], np.transpose(Vs_proj, [0,1,3,2])).transpose([0,2,1,3])
    Mu_hat_proj_mean = Mu_hat_proj.mean(axis=0)
    Mu_hat_proj_upper = np.percentile(Mu_hat_proj, 95, axis=0)
    Mu_hat_proj_lower = np.percentile(Mu_hat_proj, 5, axis=0)

    
    print('MAE on in-sample observations:')
    print('NMF:                     {}'.format(mae(Mu_nmf[...,None], Y)))
    print('Monotone NMF:            {}'.format(mae(Mu_nmf_proj[...,None], Y)))
    print('Posterior mean:          {}'.format(mae(Mu_hat_mean[...,None], Y)))
    # print('Monotone posterior mean: {}'.format(np.sqrt(mse(Mu_hat_proj_mean[...,None], Y))))
    print()


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
        print('MAE on held out observations:')
        print('NMF:                     {}'.format(mae(Mu_nmf[held_out[0], held_out[1],:,None], Y_full[held_out[0], held_out[1]])))
        print('Monotone NMF:            {}'.format(mae(Mu_nmf_proj[held_out[0], held_out[1],:,None], Y_full[held_out[0], held_out[1]])))
        print('Posterior mean:          {}'.format(mae(Mu_hat_mean[held_out[0], held_out[1],:,None], Y_full[held_out[0], held_out[1]])))
        # print('Monotone posterior mean: {}'.format(np.sqrt(mse(Mu_hat_proj_mean[held_out[0], held_out[1],:,None], Y_full[held_out[0], held_out[1]]))))
        print()

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
    if Us is not None:
        np.save(os.path.join(args.outdir, 'btf_u'), Us)
    if args.nholdout > 0:
        np.save(os.path.join(args.outdir, 'held_out'), held_out)
    print()

    if args.plot:
        print('Plotting results')
        import matplotlib.pyplot as plt
        import seaborn as sns
        Y = Y_full # Plot the full dataset
        if not os.path.exists(args.plotdir):
            os.makedirs(args.plotdir)
        if args.big_plot:
            fig, axarr = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows), sharex=True, sharey=True)
        else:
            plt.figure()
        for i in range(nrows):
            print('Row {}/{}'.format(i+1,nrows))
            for j in range(ncols):
                if args.big_plot:
                    ax = axarr[i,j]
                else:
                    ax = plt.gca()
                ax.axhline(1, color='darkgray', alpha=0.5)
                ax.plot(X, Mu_init[i,j], color='blue', label='NMF')
                if model.Mu_ep is not None:
                    ax.errorbar(X, model.Mu_ep[i,j], yerr=model.Sigma_ep[i,j], alpha=0.5)
                if len(Y.shape) > 3:
                    for k in range(ndepth):
                        ax.scatter(np.full(Y.shape[-1],X[k]), Y[i,j,k], color='black')
                else:
                    ax.scatter(X, Y[i,j], color='black')
                ax.set_ylim([0, np.nanmax(Y)+0.01])
                ax.set_xlim([X[0]-0.5,X[-1]+0.5])
                ax.plot(X, Mu_hat_mean[i,j], color='orange')
                ax.fill_between(X, Mu_hat_lower[i,j], Mu_hat_upper[i,j], color='orange', alpha=0.6)

                # Posterior predictives
                Y_hat = np.array([likelihood.sample(Mu_hat[None,i,j], size=[100,Mu_hat.shape[0],ndepth])]).reshape((-1,ndepth))
                Y_hat_upper = np.percentile(Y_hat, 95, axis=0)
                Y_hat_lower = np.percentile(Y_hat, 5, axis=0)
                ax.fill_between(X, Y_hat_lower, Y_hat_upper, color='orange', alpha=0.3)
                if args.nholdout > 0:
                    if np.any((held_out[0] == i) & (held_out[1] == j)):
                        ax.axvspan(X[0]-0.5, X[-1]+0.5, color='gray', alpha=0.3)
                if not args.big_plot:
                    plt.savefig(os.path.join(args.plotdir, 'sample{}-drug{}.pdf'.format(i,j)), bbox_inches='tight')
                    plt.close()
        if args.big_plot:
            plt.savefig(os.path.join(args.plotdir, 'all.pdf'), bbox_inches='tight')

    # Clean up shared memory and shut down worker processes
    model.shutdown()
    clean_up(args.sharedprefix)

