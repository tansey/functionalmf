import numpy as np
from utils import flatten, ilogit


if __name__ == '__main__':
    import os
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt

    '''
    Standard setup:
     
    '''
    parser = argparse.ArgumentParser(description='Generates simulated data for drug response modeling.')
    
    # General settings
    parser.add_argument('--k', type=int, default=3, help='Size of the embedding dimension.')
    parser.add_argument('--n', type=int, default=10, help='Number of simulated cell lines.')
    parser.add_argument('--m', type=int, default=11, help='Number of simulated drugs.')
    parser.add_argument('--t', type=int, default=9, help='Number of simulated concentration levels.')
    parser.add_argument('--r', type=int, default=6, help='Number of replicates per (n,m,t).')
    parser.add_argument('--seed', type=int, default=42, help='The pseudo-random number generator seed.')
    
    # Get the arguments from the command line
    args = parser.parse_args()

    # Seed the random number generator so we get reproducible results
    np.random.seed(args.seed)
    
    # Draw the cell line embeddings
    W = np.random.gamma(3,1,size=(args.n, args.k))

    # Draw the drug embeddings
    V = np.cumsum((np.random.random(size=(args.m, args.t, 1)) <= np.linspace(0.05, 0.5, args.t)[None,:,None]) * np.random.gamma(1,0.15,size=(args.m, args.t, args.k)), axis=1)

    # Calculate the ground truth effect tensor
    effects = ilogit(-(W[:,None,None] * V[None,:,:]).sum(axis=-1) + 3)

    # Draw random cell counts hierarchically
    means = np.random.normal(1, 0.1, size=(args.n,args.m,args.t+1,1))
    # means[:,:,0] = 10000
    scales = np.exp(np.random.normal(-7, 1, size=means.shape))
    shapes = means / scales
    obs = np.random.gamma(shapes, scales, size=(args.n, args.m, args.t+1, args.r))

    # Apply the drug to each population of cells
    obs[:,:,1:] *= effects[...,None]

    # Imaginary concentration levels to match the real experiments
    concentrations = np.concatenate([[-10], np.linspace(-9.12, -5.3, args.t)])

    # Save the results
    np.save('doseresponse/data/sim/obs', obs)
    np.save('doseresponse/data/sim/truth', effects)
    np.save('doseresponse/data/sim/w', W)
    np.save('doseresponse/data/sim/v', V)
    import csv
    with open('doseresponse/data/sim/data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['cell line','drug','concentration','outcome'])
        for i in range(args.n):
            for j in range(args.m):
                for t in range(args.t+1):
                    for r in range(args.r):
                        writer.writerow(['Tumor{}'.format(i),
                                         'Drug{}'.format(j),
                                         '' if t == 0 else '{:.2f}'.format(concentrations[t]),
                                         obs[i,j,t,r]])

    print('Reloading data and performing empirical Bayes likelihood estimate')
    from utils import load_data_as_pandas
    from empirical_bayes import estimate_likelihood
    df = load_data_as_pandas('doseresponse/data/sim/data.csv')
    Y, likelihood, cells, drugs, _, control_obs = estimate_likelihood(df, nbins=20, tensor_outcomes=True, plot=True)

    # Plot all the data and the truth
    print('Plotting true effects')
    for i in range(args.n):
        print('Row {}/{}'.format(i+1, args.n))
        fig, axarr = plt.subplots(args.m // 4 + min(1,args.m % 4),4, figsize=(16,12), sharey=True, sharex=True)
        for j in range(args.m):
            ax = axarr[j // 4, j % 4]
            ax.plot(concentrations, [1] + list(effects[i,j]), label='True effects', color='black')
            ax.axhline(1, color='red', ls='--', lw=3, label='Control mean')
            ax.scatter(flatten([[c]*args.r for c in concentrations[1:]]), Y[i,j].flatten(), label='Observations', color='gray')
            ax.set_title('Drug {}'.format(j))
            ax.set_ylim([0, Y.max()+1e-4])
            if j == 0:
                ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('doseresponse/plots/sim/effect-{}.pdf'.format(i), bbox_inches='tight')
        plt.close()

    




