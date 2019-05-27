'''
Baseline comparison script for Poisson-Gamma dynamical systems.

@incollection{schein:etal:poisson-gamma-ds,
title = {Poisson-Gamma dynamical systems},
author = {Schein, Aaron and Wallach, Hanna and Zhou, Mingyuan},
booktitle = {Advances in Neural Information Processing Systems},
year = {2016}
}

http://papers.nips.cc/paper/6083-poisson-gamma-dynamical-systems.pdf
'''
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import seaborn as sns
from apf.models.pgds import PGDS

def fit_pgds(Y, nembeds, # Number of embeddings to use in the decomposition
                gam = 10, # See paper. Total mass of the gamma process weights.
                tau = 1., # See paper. Concentration parameter.
                eps = 0.1, # See paper. Defines the noninformative gamma prior.
                stationary = False, # If true, uses a global shrinkage term; false shrinks locally.
                binary = False, # Set this to True if your data is binary. Otherwise, False for count data.
                nthreads = 3, # None will grab the max number of available threads for parallelization.
                time_mode=2, # Dimension of the tensor that should be smoothed in time
                nburn=5000,
                nthin=10,
                nsamples=500,
                seed=42,
                verbose = 0):
    mask = np.isnan(Y).astype(int)
    data = np.ma.array(Y, mask=mask)
    core_shp = (nembeds,)
    data_shp = data.shape

    nrows, ncols, ndepth = Y.shape[:3]

    model = PGDS(data_shp=data_shp,
             core_shp=core_shp,
             time_mode=time_mode,
             stationary=stationary,
             gam=gam, 
             tau=tau, 
             eps=eps,
             binary=binary, 
             seed=seed,
             n_threads=nthreads)
    Mu = np.zeros((nsamples, nrows, ncols, ndepth))
    W = np.zeros((nsamples, nrows, nembeds))
    V = np.zeros((nsamples, ncols, nembeds))
    U = np.zeros((nsamples, ndepth, nembeds))
    for step in range(nsamples+1):
        model.fit(data,
              n_itns=nthin if step > 0 else nburn,          # how many MCMC iterations
              initialize=step == 0,  # whether to initialize
              verbose=verbose,             # how often to printout a state (set verbose=0 for silent)
              impute_after=0,         # when to start imputing missing data (if applicable)
              schedule={},            # when and how often to update each variable 
              fix_state={},           # a dict containing values of parameters to clamp
              init_state={})          # an initial state for parameters

        if step > 0:
            state = dict(model.get_state())
            W[step-1], V[step-1], U[step-1] = [z.T for z in get_matrices(state, data_shp)]
            Mu[step-1] = model.reconstruct()

    # Get the expected means
    # Mu = np.einsum('nik,njk,ntk->nijt', W, V, U)
    return Mu, (W, V, U)

def get_matrices(state, data_shp):
    mtx_MKD = state['mtx_MKD']
    for mode, D in enumerate(data_shp):
        yield mtx_MKD[mode][:, :D]



