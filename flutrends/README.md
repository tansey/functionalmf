# Google Flu Trends
This code runs the benchmark to compare predictive performance on the Google
Flu Trends dataset from Fox and Dunson (2015). To run everything:

```
# Build the state-level dataset and hold out some random years
python create_datasets.py

# Run the BNP Covariance Regression code from Fox and Dunson (2015)
matlab runstuff_varinds_flu_stats.m

# Run the Gaussian BTF model + benchmark performance against BNP-CovReg
python benchmark.py
```

See below for copyright informaton copied from the original Fox and Dunson (2015) code.
Modifications:

- Cut out all the unnecessary simulation experiments

- Original code did not match their paper. Their paper claims to work on log-counts,
but the original code release worked on raw counts (technically it worked with
`raw / max(var(regions)) * 1.7`, for some reason). The code here works with
raw counts as reported in the paper.

- Works with just the 50 states instead of all 183 regions. This is a more challenging
problem because city-level information is highly informative of state counts.



===============================================================================
Matlab Software for Bayesian Nonparametric Covariance Regression
===============================================================================

Copyright (C) 2011, Emily B. Fox
(fox[at]stat[dot]duke[dot]edu)

This software package includes Matlab scripts that implement the Gibbs sampling
algorithm for the model described in:
  Bayesian Nonparametric Covariance Regression
  E. B. Fox and D. B. Dunson
  arXiv:1101.2017, January 2011 (revised February 2011).
  http://arxiv.org/abs/1101.2017
Please cite this paper in any publications using the HDP-AR-HMM or HDP-SLDS package.

========================================================================
Package Organization and Documentation
========================================================================

Summary of BNP Covariance Regression package contents:

BNP_covreg.m:
  Main inference script for running the sampler without missing data 
  or imputing the missing values.
BNP_covreg_varinds.m:
  Main inference script for running the sampler with analytic marginalization
  of missing values.
runstuff_BNPcovreg.m and runstuff_var_inds_flu.m: see below for explanation

/utilities:  
  Script SIMplots.m contains an example of how to process and visualize
  the results from the stored samples.

========================================================================
Setup and Usage Examples
========================================================================

For an example of running the sampler without missing data or imputing
the missing values, see runstuff_BNPcovreg.m.  This script examines a 
synthetic data example.

For an example of running the sampler with analytic marginalization of 
missing values, see runstuff_varinds_flu.m.  This script examines the 
Google Flu Trends dataset.

========================================================================
Copyright & License
========================================================================

Copyright (C) 2011, Emily B. Fox.

http://web.mit.edu/ebfox/www/

Permission is granted for anyone to copy, use, or modify these
programs and accompanying documents for purposes of research or
education, provided this copyright notice is retained, and note is
made of any changes that have been made.

These programs and documents are distributed without any warranty,
express or implied.  As the programs were written for research
purposes only, they have not been tested to the degree that would be
advisable in any important application.  All use of these programs is
entirely at the user's own risk.

