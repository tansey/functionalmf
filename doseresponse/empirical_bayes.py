import numpy as np
from scipy.stats import gamma
from scipy.misc import logsumexp

class GammaGridLikelihood:
    def __init__(self, mean_grid, mean_probs, variance):
        self.shape_grid = mean_grid**2 / variance
        self.scale_grid = variance / mean_grid
        self.probs_grid = mean_probs
        
    def logpdf(self, y, effect):
        assert len(y.shape) > 1
        scales = self.scale_grid[None]
        shapes = self.shape_grid[None]
        probs = self.probs_grid
        while len(scales.shape) <= len(y.shape):
            scales = scales[None]
            shapes = shapes[None]
            probs = probs[None]
        y = y[...,None]
        effect = effect[...,None]
        # Probability given shape/scale
        component_logprobs = np.nansum(gamma.logpdf(y, shapes, scale=scales*effect), axis=-2)
        # Log-prob of the mixture model
        result = logsumexp(component_logprobs, b=probs, axis=-1)
        # Total probs over mixture
        return result

    def sample(self, effect, size=1):
        idx = np.random.choice(self.probs_grid.shape[0], size=size)
        shapes, scales = self.shape_grid[idx], self.scale_grid[idx]
        return np.random.gamma(shapes, scales*effect)


def estimate_likelihood(df, nbins=50, control_mean=1, tensor_outcomes=False, plot=False):
    from collections import defaultdict
    cells = list(df['cell line'].unique())
    drugs = list(df['drug'].unique())
    concentrations = [c for c in sorted(df['concentration'].unique()) if not np.isnan(c)]
    print('Concentration values:', concentrations)
    outcomes = defaultdict(list)
    controls = defaultdict(list)
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(idx)
        cell = cells.index(row['cell line'])
        drug = drugs.index(row['drug'])
        conc = row['concentration']
        outcome = row['outcome']
        if np.isnan(conc):
            controls[(cell, drug)].append(outcome)
        else:
            outcomes[(cell, drug, concentrations.index(conc))].append(outcome)

    # Estimate the gamma parameters for the controls, after shrinking to reasonable sizes
    for cell in range(len(cells)):
        for drug in range(len(drugs)):
            if (cell,drug) not in controls:
                continue
            obs = controls[(cell,drug)]
            mu, std = np.mean(obs), np.std(obs)
            for t in range(len(concentrations)):
                outcomes[(cell, drug, t)] = [o * control_mean / mu for o in outcomes[(cell, drug, t)]]
            obs = [o * control_mean / mu for o in controls[(cell, drug)]]
            mu, std = np.mean(obs), np.std(obs)
            controls[(cell, drug)] = obs

    # Empirical Bayes estimate of cell count prior parameters
    means = []
    noise = []
    for cell in range(len(cells)):
        for drug in range(len(drugs)):
            if (cell,drug) not in controls:
                continue
            obs0 = controls[(cell,drug)] # control observations
            obs1 = outcomes[(cell, drug, 0)] # first dosage observation
            if len(obs1) > 0 and np.mean(obs1) > control_mean:
                # Look at cases where first dosage is higher than control
                # to estimate variation in cell concentration
                # NOTE: this is under-estimating variance since the
                # first dosage may have some small effect even when the
                # mean is higher than the control
                means.append(np.mean(obs1))    
            # Track the distribution of noise in the controls to estimate
            # observation noise
            noise.extend((np.array(obs0) - control_mean)**2)
    means, noise = np.array(means), np.array(noise).mean()


    import matplotlib
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import seaborn as sns
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        # Histogram poisson regression for estimation of the mean distribution
        counts, bins, _ = plt.hist(means, bins=nbins//2) # Use N/2 bins because we only look at the right halfspace
        K = 3 # K'th-order polynomial poisson regression
        X = np.array([np.arange(len(counts))**k for k in range(K+1)]).T
        poisson_glm = sm.GLM(counts, X, family=sm.families.Poisson())
        poisson_glm_results = poisson_glm.fit()

        # Assume the mean prior distribution is symmetric
        mean_grid = np.concatenate([2*control_mean-(bins[:-1] + bins[1:])[::-1]/2, (bins[:-1] + bins[1:])/2])
        mean_probs = np.concatenate([poisson_glm_results.fittedvalues[::-1], poisson_glm_results.fittedvalues])
        mean_probs /= mean_probs.sum() # normalize to make a proper distribution
        if plot:
            plt.plot(mean_grid, mean_probs*2*counts.sum(), lw=3, label='Empirical Bayes prior')
            plt.xlabel('Initial cell population size', fontsize=18, weight='bold', fontname='Times New Roman')
            plt.ylabel('# observations', fontsize=18, weight='bold', fontname='Times New Roman')
            plt.legend(loc='upper right')
            plt.savefig('plots/empirical-bayes-means.pdf', bbox_inches='tight')
        plt.close()

    # Create the mixture likelihood
    likelihood = GammaGridLikelihood(mean_grid, mean_probs, noise)
    
    # Convert the observations to a nan tensor
    if tensor_outcomes:
        max_replicates = max([len(o) for o in outcomes.values()])
        Y = np.full((len(cells), len(drugs), len(concentrations), max_replicates), np.nan)
        for (i,j,t), o in outcomes.items():
            for r, o_r in enumerate(o):
                Y[i,j,t,r] = o_r
        outcomes = Y

    return outcomes, likelihood, cells, drugs, concentrations, controls






