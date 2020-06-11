'''
Code to setup the data for the Google Flu Trends benchmark.

Run this before running runstuff_varinds_flu_states.m.
'''

import matplotlib.dates as mdates
import datetime
import numpy as np
from scipy.io import loadmat, savemat

np.random.seed(42)

# Focus on just the state-level data
df_states = loadmat('flu_US.mat')
Y_states = df_states['data'][:,1:51]
df_states['data'] = Y_states
df_states['USnames'] = df_states['USnames'][1:51]
df_states['USnames_test'] = df_states['USnames_test'][1:51]
savemat('flu_US_states.mat', df_states)

# Hold out some years
years = np.array([int(x[0][0][:4]) for x in df_states['dates']])
has_week = ~np.isnan(df_states['data'])
state_idx, year_start, year_end = [], [], []
for yr in range(years.min(), years.max()+1):
    has_year = np.any(has_week[years == yr], axis=0)
    state_idx.extend(np.arange(df_states['data'].shape[1])[has_year])
    year_start.extend([np.arange(years.shape[0])[years == yr][0]]*has_year.sum())
    year_end.extend([np.arange(years.shape[0])[years == yr][-1]+1]*has_year.sum())
indices = np.array([state_idx, year_start, year_end]).T
to_hold = indices[np.random.choice(indices.shape[0], replace=False, size=int(np.ceil(indices.shape[0]*0.1)))]
for i,j,k in to_hold:
    df_states['data'][j:k,i] = np.nan
savemat('flu_US_states_train.mat', df_states)
np.save('held_out_years', to_hold)