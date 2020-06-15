import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




if __name__ == '__main__':
    import os
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    parser = argparse.ArgumentParser(description='Plots important features for drug sensitivity and resistance.')
    
    # General settings
    parser.add_argument('--data', default='doseresponse/data/sim/data.csv', help='Location of the data file.')
    parser.add_argument('--features', default='doseresponse/data/sim/features.csv', help='Matrix of binary features for each row.')
    parser.add_argument('--outdir', default='doseresponse/data/sim/', help='Directory where all results will be saved.')
    parser.add_argument('--plot', action='store_true', help='If true, the data and results will be plotted at the end.')
    parser.add_argument('--big_plot', action='store_true', help='If true and plot is true, a single huge plot will be made.')
    parser.add_argument('--plotdir', default='doseresponse/plots/sim/', help='Directory where all results will be saved.')
    parser.add_argument('--ntop', type=int, default=20, help='The top sensitivity and resistance samples to report.')

    # Get the arguments from the command line
    args = parser.parse_args()

    # Load the drug, cell line, and feature labels
    drugs = np.load(os.path.join(args.outdir, 'drugs.npy'))
    cells = np.load(os.path.join(args.outdir, 'cells.npy'))
    features = pd.read_csv(args.features, index_col=0, header=0).columns.values

    # Load the embeddings
    Ws = np.load(os.path.join(args.outdir, 'btf_w.npy'))
    Vs = np.load(os.path.join(args.outdir, 'btf_v.npy'))
    Us = np.load(os.path.join(args.outdir, 'btf_u.npy'))

    # Calculate the summary statistics
    feature_probs = np.einsum('znk,zmk->znm', Ws, Us).mean(axis=0)
    auc_scores = np.trapz(np.einsum('znk,zmtk->znmt', Ws, Vs), dx=1/(Vs.shape[-1]-1), axis=-1).mean(axis=0)

    # Fit regression lines to each feature and auc score
    from scipy.stats import linregress
    index = []
    feature_fits = []
    for fname, x in zip(features, feature_probs.T):
        for dname, y in zip(drugs, auc_scores.T):
            index.append((fname, dname))
            feature_fits.append(linregress(x,y))

    feature_fits = pd.DataFrame(feature_fits,
                                index=index,
                                columns=['slope', 'intercept', 'r-value', 'p-value', 'stderr'])

    print(feature_fits.describe())

    print('Top {} resistant:'.format(args.ntop))
    print(feature_fits.iloc[np.argsort(feature_fits['r-value'].values)[-args.ntop:][::-1]])
    print()
    print('Top {} sensitive:'.format(args.ntop))
    print(feature_fits.iloc[np.argsort(feature_fits['r-value'].values)[:args.ntop]])
    print()

    if args.plot:
        # Plot each hit
        if not os.path.exists(args.plotdir):
            os.makedirs(args.plotdir)
        for idx in np.concatenate([np.argsort(feature_fits['r-value'].values)[-args.ntop:][::-1], np.argsort(feature_fits['r-value'].values)[:args.ntop]]):
            with sns.axes_style('white'):
                plt.rc('font', weight='bold')
                plt.rc('grid', lw=3)
                plt.rc('lines', lw=3)
                matplotlib.rcParams['pdf.fonttype'] = 42
                matplotlib.rcParams['ps.fonttype'] = 42

                # Get the indices for the AUC scores and the feature values
                fidx, didx = idx // auc_scores.shape[1], idx % auc_scores.shape[1]
                slope, intercept, r_value, p_value, stderr = feature_fits.iloc[idx]
                feature, drug = feature_fits.index[idx]

                plt.scatter(feature_probs[:,fidx], auc_scores[:,didx], color='gray', alpha=0.5)
                plt.plot([0,1], [intercept, intercept+slope], color='red', lw=3, label='$r^2={:.2f}$'.format(r_value**2))
                plt.xlabel('Biomarker probability', fontsize=18, weight='bold')
                plt.ylabel('Dose-response AUC', fontsize=18, weight='bold')
                plt.xlim([0,1])
                plt.ylim([0,1])

                legend_props = {'weight': 'bold', 'size': 14}
                plt.legend(loc='upper right', prop=legend_props)
                plt.title('{} + {}'.format(feature, drug), fontsize=22, weight='bold')
                plt.savefig(os.path.join(args.plotdir, 'feature-importance-{}.pdf'.format(idx)), bbox_inches='tight')
                plt.close()












