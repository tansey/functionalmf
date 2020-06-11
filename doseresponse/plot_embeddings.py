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
    parser.add_argument('--basedir', default='doseresponse/data/sim/', help='Directory where all results will be saved.')
    parser.add_argument('--plotdir', default='doseresponse/plots/sim/', help='Directory where all results will be saved.')
    parser.add_argument('--big_plot', action='store_true', help='If true and plot is true, a single huge plot will be made.')
    parser.add_argument('--seed', type=int, default=4, help='The pseudo-random number generator seed.')
    parser.add_argument('--truth', action='store_true', help='If true, this is simulated data with ground truth known.')
    parser.add_argument('--features', action='store_true', help='If specified, plots each embedding colored by each feature. Generates N plots for N features.')
    parser.add_argument('--reducer', default='tsne', help='Which dimensionality reducer to use.')

    # Get the arguments from the command line
    args = parser.parse_args()

    # Load the data
    df = load_data_as_pandas(os.path.join(args.basedir, 'data.csv'))
    resultsdir = os.path.join(args.basedir, 'seed{}'.format(args.seed))
    # W = np.load(os.path.join(resultsdir, 'btf_w.npy'))
    W = np.load(os.path.join(args.basedir, 'nmf_w.npy')) # TEMP
    V = np.load(os.path.join(args.basedir, 'nmf_v.npy')) # TEMP
    print(W.shape, V.shape)
    Mu = (W[:,None,None] * V[None]).sum(axis=-1)
    # Mu = np.load(os.path.join(resultsdir, 'btf.npy'))

    # Load the cell names
    cells = np.load(os.path.join(args.basedir, 'cells.npy'))

    # Calculate the AUC via the trapezoid rule
    AUC = np.trapz(Mu)
    W = AUC

    if len(W.shape) == 3:
        print('Taking the Bayes estimate of embeddings')
        W = W.mean(axis=0)
        # W = W[-1,:,-2:]

    # If the embeddings are not 2d, project to 2d via MDS
    if W.shape[1] != 2:
        print('Projecting down to 2 dimensions')
        if args.reducer == 'umap':
            import umap
            reducer = umap.UMAP()
        elif args.reducer == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2)
        elif args.reducer == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        W = reducer.fit_transform(W)

    plt.scatter(W[:,0], W[:,1])
    plt.savefig(os.path.join(args.plotdir, 'embeddings.pdf'), bbox_inches='tight')
    plt.close()


    if args.features:
        import pandas as pd
        df = pd.read_csv(os.path.join(args.basedir, 'features.csv'), index_col=0)

        # for i in range(df.shape[0]):
        i = df.shape[0] - 1
        print(df.index[i])
        feats = df.iloc[i]
        featname = df.index[i].replace(' ', '').replace('(','').replace(')', '').replace('\'', '').replace(',', ' in ')
        labels = [feats[name] if name in feats else 'Unknown' for name in cells]

        # Convert binary to boolean
        palette = None
        labels = ['False' if l == '0' or l == '0.0' else ('True' if l == '1' or l == '1.0' else l) for l in labels]
        if 'False' in labels or 'True' in labels:
            labels = [l if l == 'False' or l == 'True' else 'Unknown' for l in labels]
            palette = {'True': 'orange', 'False': 'blue', 'Unknown': 'gray'}
        if 'Glioma' in labels:
            labels = ['Glioma' if l == 'Glioma' else ('Gastric' if 'Gastric' in l else 'Other') for l in labels]
            palette = {'Glioma': 'blue', 'Gastric': 'orange', 'Other': 'gray'}
            # labels = ['Brain' if l == 'Glioma' or 'brain' in l else ('Lung' if 'NSCLC' in l or 'Lung' in l else ('Gastric' if 'Gastric' in l else 'Other')) for l in labels]
            # palette = {'Brain': 'blue', 'Lung': 'orange', 'Gastric': 'green', 'Other': 'gray'}
        # print(labels)

        df_plot = pd.DataFrame({featname: labels, 'Dimension 1': W[:,0], 'Dimension 2': W[:,1]})
        sns.scatterplot(x='Dimension 1', y='Dimension 2', hue=featname, data=df_plot, palette=palette)
        plt.savefig(os.path.join(args.plotdir, 'embeddings-{}.pdf'.format(featname.replace(' in ', '-'))), bbox_inches='tight')
        plt.close()




        


