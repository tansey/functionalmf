# Example modeling count data in the GDELT dataset

The [GDELT project](https://www.gdeltproject.org/) tracks different kinds of interactions between nations. We used a subset of the dataset here to demonstrate different ways to model count data. Specifically, we focused on the `cooperate` action and the G20 nations. 
See `create_datasets.py` for details on how we preprocessed the dataset.

See `benchmark.py` for full code on setting up both the negative binomial and Poisson models. The upshot here is that a negative binomial model with Polya-Gamma augmentation fits well to in-sample curves (white entries) but is unstable for held out curves (shaded entries):

![Visualization of the Negative Binomial functional matrix factorization](https://github.com/tansey/functionalmf/raw/master/img/nb-tensor-filtering-politics.png)

On the other hand, using a non-conjugate Poisson likelihood with GASS inference is better:

![Visualization of the Poisson functional matrix factorization](https://github.com/tansey/functionalmf/raw/master/img/poisson-tensor-filtering-politics.png)

The Poisson model requires a bit more work to initialize, since it uses a Poisson tensor factorization algorithm to find a good starting point. It also mixes slower, but produces a better overall estimate in the end.

