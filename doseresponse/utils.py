import numpy as np
import pandas as pd

def load_data_as_pandas(filename):
    return pd.read_csv(filename, header=0)

def flatten(l):
    '''Takes a list of lists and flattens it.'''
    return [item for sublist in l for item in sublist]

def ilogit(x):
    return 1. / (1. + np.exp(-x))
