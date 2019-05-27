import numpy as np
from empirical_bayes import estimate_likelihood
from utils import load_data_as_pandas





if __name__ == '__main__':
    import os
    import argparse
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    parser = argparse.ArgumentParser(description='Bayesian tensor filtering for dose-response modeling.')
    
    # General settings
    parser.add_argument('--data', default='doseresponse/data/sim/data.csv', help='Location of the data file.')
    parser.add_argument('--basedir', default='doseresponse/data/sim/', help='Directory where all results will be saved.')
    parser.add_argument('--plotdir', default='doseresponse/plots/sim/', help='Directory where all results will be saved.')
    parser.add_argument('--big_plot', action='store_true', help='If true and plot is true, a single huge plot will be made.')
    parser.add_argument('--truth', help='If given, this is the location of the known true effect sizes (good for assessing simulated data).')

    # Model settings
    parser.add_argument('--nembeds', type=int, default=5, help='Size of the embedding dimension.')
    parser.add_argument('--seed', type=int, default=2, help='The pseudo-random number generator seed.')
    
    # Get the arguments from the command line
    args = parser.parse_args()

    # Load the data
    df = load_data_as_pandas(args.data)

    print('Loading data and performing empirical Bayes likelihood estimate')
    Y, likelihood, cells, drugs, concentrations, control_obs = estimate_likelihood(df, tensor_outcomes=True, plot=True)
    print(Y.shape)
    

    resultsdir = os.path.join(args.basedir, 'seed{}'.format(args.seed))
    nmf = np.load(os.path.join(resultsdir, 'nmf.npy'))
    lmf = np.load(os.path.join(resultsdir, 'logistic_mf.npy'))
    btf = np.load(os.path.join(resultsdir, 'btf.npy'))
    if os.path.exists(os.path.join(resultsdir, 'held_out.npy')):
        held_out = np.load(os.path.join(resultsdir, 'held_out.npy'))
    else:
        held_out = np.array(list(np.ndindex(Y.shape[:2])))

    if args.truth is not None:
        truth = np.load(args.truth)

    btf_mean = btf.mean(axis=0)
    btf_lower = np.percentile(btf, 5, axis=0)
    btf_upper = np.percentile(btf, 95, axis=0)

    nheld = held_out.shape[0]
    nfigrows = nheld // 4 + (nheld % 4 > 0)
    nfigcols = 4

    print('Plotting results')
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        if args.big_plot:
            fig, axarr = plt.subplots(nfigrows, nfigcols, figsize=(5*nfigcols,5*nfigrows), sharex=True, sharey=True)
        for i,(ridx, cidx) in enumerate(held_out):
            print(ridx, cidx)
            if args.big_plot:
                ax = axarr[i // nfigcols, i % nfigcols]
            else:
                ax = plt
            ax.plot(concentrations, nmf[ridx,cidx], color='blue', ls='--', label='NMF', lw=3)
            ax.plot(concentrations, lmf[ridx,cidx], color='green', ls='-.', label='LMF', lw=3)
            if len(Y.shape) > 3:
                for k in range(Y.shape[2]):
                    ax.scatter(np.full(Y.shape[-1],concentrations[k]), Y[ridx,cidx,k], color='gray')
            else:
                ax.scatter(concentrations, Y[ridx,cidx], color='gray')
            plt.ylim([0, np.nanmax(Y)+0.01])
            ax.plot(concentrations, btf_mean[ridx,cidx], color='orange', label='BTF', lw=3)
            # ax.fill_between(concentrations, btf_lower[ridx,cidx], btf_upper[ridx,cidx], color='orange', alpha=0.5)

            if args.truth is not None:
                ax.plot(concentrations, truth[ridx, cidx], color='black', label='Truth', lw=3)

            obs = likelihood.sample(1, size=(1000, len(concentrations)))
            obs *= btf[np.random.choice(btf.shape[0], size=1000), ridx, cidx]
            obs_lower = np.percentile(obs, 5, axis=0)
            obs_upper = np.percentile(obs, 95, axis=0)
            ax.fill_between(concentrations, obs_lower, obs_upper, color='orange', alpha=0.5)
            btf_lower = np.percentile(btf[:,ridx,cidx], 5, axis=0)
            btf_upper = np.percentile(btf[:,ridx,cidx], 95, axis=0)
            ax.plot(concentrations, btf_lower, ls=':', color='darkorange')
            ax.plot(concentrations, btf_upper, ls=':', color='darkorange')
            if not args.big_plot:
                legend_props = {'weight': 'bold', 'size': 14}
                plt.legend(loc='upper right', prop=legend_props)
                plt.xlabel('Log(concentration)', fontsize=18)
                plt.ylabel('Survival %', fontsize=18)
                plt.savefig(os.path.join(args.plotdir, 'seed{}-{}-{}.pdf'.format(args.seed,ridx,cidx)), bbox_inches='tight')
                plt.close()
        if args.big_plot:
            plt.savefig(os.path.join(args.plotdir, 'seed{}.pdf'.format(args.seed)), bbox_inches='tight')





