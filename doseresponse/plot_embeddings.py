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
    parser.add_argument('--outdir', default='doseresponse/data/sim/', help='Directory where all results will be saved.')
    parser.add_argument('--plotdir', default='doseresponse/plots/sim/', help='Directory where all results will be saved.')
    parser.add_argument('--id_map', default='doseresponse/plots/sim/cancer_types.csv', help='File where sample type is stored.')
    parser.add_argument('--meta', default='doseresponse/plots/sim/metadata.csv', help='File where sample type is stored.')
    parser.add_argument('--big_plot', action='store_true', help='If true and plot is true, a single huge plot will be made.')
    parser.add_argument('--seed', type=int, default=42, help='The pseudo-random number generator seed.')
    parser.add_argument('--truth', action='store_true', help='If true, this is simulated data with ground truth known.')
    parser.add_argument('--features', action='store_true', help='If specified, plots each embedding colored by each feature. Generates N plots for N features.')
    parser.add_argument('--reducer', default='pca', help='Which dimensionality reducer to use.')

    # Get the arguments from the command line
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load the data
    Ws = np.load(os.path.join(args.outdir, 'btf_w.npy'))
    Vs = np.load(os.path.join(args.outdir, 'btf_v.npy'))
    Us = np.load(os.path.join(args.outdir, 'btf_u.npy'))
    W = Ws
    # W = np.trapz(np.einsum('znk,zmtk->znmt', Ws, Vs), dx=1/(Vs.shape[-1]-1), axis=-1).mean(axis=0)
    # W = np.einsum('znk,zmk->znm', Ws, Us).mean(axis=0)
    if len(W.shape) == 3:
        # print('Taking the mean estimate of embeddings')
        # W = W.mean(axis=0)
        print('Taking the last sample as embeddings')
        W = W[-1]
    print(W.shape)
    # If the embeddings are not 2d, project to 2d
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
        elif args.reducer == 'first':
            class Reducer:
                def fit_transform(self, x):
                    return x[:,:2]
            reducer = Reducer()
        W = reducer.fit_transform(W)

    # Load the cell names
    cells = np.load(os.path.join(args.outdir, 'cells.npy'))


    UIDs = pd.read_csv(args.id_map, header=0)
    meta = pd.read_csv(args.meta, index_col=0, header=0)
    groups = {c: meta.loc[uid]['Group'] for c, uid in zip(UIDs['Internal ID'], UIDs['Sample ID']) if uid in meta.index}

    # Look at stage of the cancer
    def get_stage(s):
        if 'IV' in s:
            return 'IV'
        elif 'III' in s:
            return 'III'
        elif 'II' in s:
            return 'II'
        elif 'I' in s:
            return 'I'
        return 'Unknown'
    stages = {c: (get_stage(meta.loc[uid]['Stage']) if uid in meta.index else 'Unknown') for c, uid in zip(UIDs['Internal ID'], UIDs['Sample ID'])}
    stage_colors = {'Unknown': 'lightgray', 'I': 'green', 'II': 'orange', 'III': 'purple', 'IV': 'blue'}
    W_colors = [stage_colors[stages[c]] if c in stages else '0' for c in cells]
    
    # Look at sex of the patient
    sexes = {c: (meta.loc[uid]['Sex'] if uid in meta.index else 'Unknown') for c, uid in zip(UIDs['Internal ID'], UIDs['Sample ID'])}
    sex_colors = {'Unknown': 'lightgray', 'M': 'blue', 'F': 'red'}
    W_colors = [sex_colors[sexes[c]] if c in sexes else 'lightgray' for c in cells]

    # Look at age of the patient
    ages = {c: (meta.loc[uid]['Age'] if uid in meta.index else 'Unknown') for c, uid in zip(UIDs['Internal ID'], UIDs['Sample ID'])}
    W_colors = [ages[c] if c in ages else np.nan for c in cells]
    
    # from collections import defaultdict
    # cell_type_map = defaultdict(list)
    # for c,w in zip(cells, W):
    #     if c not in groups:
    #         continue
    #     cell_type_map[groups[c]].append(w)
    # cell_type_map = {t: w for t,w in cell_type_map.items() if len(w) > 10}
    
    prefixes = np.unique([c[:3].replace('1','') for c in cells])
    cell_type_map = {t: [] for t in prefixes}
    color_map = {t: [] for t in prefixes}
    assert len(cells) == len(W)
    for c,w,color in zip(cells, W, W_colors):
        prefix = c[:3].replace('1','')
        cell_type_map[prefix].append(w)
        color_map[prefix].append(color)

    filtered_colors = {t: c for t,c in color_map.items() if len(c) >= 10}
    filtered_map = {t: w for t,w in cell_type_map.items() if len(w) >= 10}
    m = {'BT': 'Brain', 'GBM': 'Brain', 'LC': 'Lung', 'MBT': 'Brain', 'RCC': 'Liver', 'OC': 'Ovarian', 'BCA': 'Breast', 'GC': 'Gastric'}
    from collections import defaultdict
    cell_type_map = defaultdict(list)
    color_map = defaultdict(list)
    for t,w in filtered_map.items():
        label = m[t]
        cell_type_map[label].extend(w)
        color_map[label].extend(filtered_colors[t])

    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        ncols = (np.arange(5)+1)[(len(cell_type_map) % (np.arange(5)+1)) == 0].max()
        nrows = int(np.ceil(len(cell_type_map) / ncols))
        fig, axarr = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), sharex=True, sharey=True)
        for idx, (t,w) in enumerate(cell_type_map.items()):
            ax = axarr[idx // ncols, idx % ncols]
            W_t = np.array(w)
            ax.scatter(W_t[:,0], W_t[:,1], c='black')
            ax.set_title(t, fontsize=22, weight='bold')
            ax.set_xlabel('PC1', fontsize=18, weight='bold')
            ax.set_ylabel('PC2', fontsize=18, weight='bold')
        # plt.legend(loc='lower left', ncol=2)
        plt.savefig(os.path.join(args.plotdir, 'embeddings.pdf'), bbox_inches='tight')
        plt.close()


    # if args.features:
    #     import pandas as pd
    #     df = pd.read_csv(os.path.join(args.basedir, 'features.csv'), index_col=0)

    #     # for i in range(df.shape[0]):
    #     i = df.shape[0] - 1
    #     print(df.index[i])
    #     feats = df.iloc[i]
    #     featname = df.index[i].replace(' ', '').replace('(','').replace(')', '').replace('\'', '').replace(',', ' in ')
    #     labels = [feats[name] if name in feats else 'Unknown' for name in cells]

    #     # Convert binary to boolean
    #     palette = None
    #     labels = ['False' if l == '0' or l == '0.0' else ('True' if l == '1' or l == '1.0' else l) for l in labels]
    #     if 'False' in labels or 'True' in labels:
    #         labels = [l if l == 'False' or l == 'True' else 'Unknown' for l in labels]
    #         palette = {'True': 'orange', 'False': 'blue', 'Unknown': 'gray'}
    #     if 'Glioma' in labels:
    #         labels = ['Glioma' if l == 'Glioma' else ('Gastric' if 'Gastric' in l else 'Other') for l in labels]
    #         palette = {'Glioma': 'blue', 'Gastric': 'orange', 'Other': 'gray'}
    #         # labels = ['Brain' if l == 'Glioma' or 'brain' in l else ('Lung' if 'NSCLC' in l or 'Lung' in l else ('Gastric' if 'Gastric' in l else 'Other')) for l in labels]
    #         # palette = {'Brain': 'blue', 'Lung': 'orange', 'Gastric': 'green', 'Other': 'gray'}
    #     # print(labels)

    #     df_plot = pd.DataFrame({featname: labels, 'Dimension 1': W[:,0], 'Dimension 2': W[:,1]})
    #     sns.scatterplot(x='Dimension 1', y='Dimension 2', hue=featname, data=df_plot, palette=palette)
    #     plt.savefig(os.path.join(args.plotdir, 'embeddings-{}.pdf'.format(featname.replace(' in ', '-'))), bbox_inches='tight')
    #     plt.close()




        


