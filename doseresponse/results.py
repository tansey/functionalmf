import numpy as np
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from empirical_bayes import estimate_likelihood
from utils import load_data_as_pandas
from functionalmf.utils import mse, mae

if __name__ == '__main__':
    models = [{'name': 'NMF', 'file': 'nmf.npy', 'preprocess': lambda x: x},
                {'name': 'Logistic MF', 'file': 'logistic_mf.npy', 'preprocess': lambda x: x},
                {'name': 'BTF', 'file': 'btf.npy', 'preprocess': lambda x: x.mean(axis=0)},
                {'name': 'Monotone NMF', 'file': 'nmf_mono.npy', 'preprocess': lambda x: x},
                {'name': 'Monotone BTF', 'file': 'btf_mono.npy', 'preprocess': lambda x: x.mean(axis=0)}]
    nmodels = len(models)


    metrics = [{'name': 'MAE',  'fun': lambda Y, Mu, pred: mae(Y, pred[...,None])},
               {'name': 'RMSE', 'fun': lambda Y, Mu, pred: np.sqrt(mse(Y, pred[...,None]))},
               {'name': 'NLL', 'fun': lambda Y, Mu, pred: -np.nansum(likelihood.logpdf(Y, pred[...,None]))}]
    nmetrics = len(metrics)

    parser = argparse.ArgumentParser(description='Results for Bayesian tensor filtering for dose-response modeling.')
    
    # General settings
    parser.add_argument('seeds', nargs='+', help='The random seeds of the different trials to process.')
    parser.add_argument('--data', default='doseresponse/data/sim/data.csv', help='Location of the data file.')
    parser.add_argument('--outdir', default='doseresponse/data/sim/', help='Directory where all results will be saved.')
    parser.add_argument('--latex', action='store_true', help='Output a latex table at the end.')
    parser.add_argument('--truth', help='If given, this is the location of the known true effect sizes (good for assessing simulated data).')
    
    # Get the arguments from the command line
    args = parser.parse_args()

    ntrials = len(args.seeds)

    # Load the data
    df = load_data_as_pandas(args.data)

    print('Loading data and performing empirical Bayes likelihood estimate')
    Y, likelihood, cells, drugs, concentrations, control_obs = estimate_likelihood(df, tensor_outcomes=True, plot=False)

    if args.truth is not None:
        truth = np.load(args.truth)
        metrics.append({'name': 'MAE (truth)', 'fun': lambda Y, Mu, pred: mae(Mu, pred)})
        metrics.append({'name': 'RMSE (truth)', 'fun': lambda Y, Mu, pred: np.sqrt(mse(Mu, pred))})
        nmetrics = len(metrics)

    print('Collecting results')
    results = np.zeros((ntrials, nmetrics, nmodels))
    for trial, seed in enumerate(args.seeds):
        cur_dir = os.path.join(args.outdir, 'seed{}'.format(seed))
        if os.path.exists(os.path.join(cur_dir, 'held_out.npy')):
            held_out = np.load(os.path.join(cur_dir, 'held_out.npy'))
        else:
            held_out = np.array(list(np.ndindex(Y.shape[:2])))

        Y_test = Y[held_out[0], held_out[1]]
        predictions = [m['preprocess'](np.load(os.path.join(cur_dir, m['file'])))[held_out[0], held_out[1]] for m in models]
        Mu_test = truth[held_out[0], held_out[1]] if args.truth is not None else None
        
        for metidx, metric in enumerate(metrics):
            results[trial, metidx] = [metric['fun'](Y_test, Mu_test, pred) for pred in predictions]

    print('Final results:')
    print(('{:<20}'*(nmetrics+1)).format(*(['Model'] + [m['name'] for m in metrics])))
    for i, model in enumerate(models):
        model_results = ''.join(['{:<20}'.format('{:.2f} +/- {:.2f}'.format(r,s)) for r,s in zip(results[:,:,i].mean(axis=0), results[:,:,i].std(axis=0)/np.sqrt(results.shape[0]))])
        print('{:<20}'.format(model['name']) + model_results)

    if args.latex:
        print('Latex table:')
        print('\\begin{tabular}{' + 'l' + 'c'*nmetrics + '}')
        print(' & '.join(['Model'] + [m['name'] for m in metrics]), ' \\\\ \\hline')
        mean_results = results.mean(axis=0)
        best_models = [np.argmin(r) for r in mean_results]
        for i, model in enumerate(models):
            print(' & '.join([model['name']] + [('{:.2f}'.format(r) if b != i else '\\textbf{' + '{:.2f}'.format(r) + '}') for r, b in zip(mean_results[:,i], best_models)]), ' \\\\')
        print('\\end{tabular}')












