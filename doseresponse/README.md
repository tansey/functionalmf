# Dose-response modeling in cancer drug studies

This is an example of dose-response modeling for a multi-drug, multi-tumor exploratory cancer drug study. The model assumes the experiments are conducted on microwell plates where each plate tests one drug on one sample, with the microwells used for different concentrations of the drug and for replicates.


## Experimental design
The experimental design (microwell plate layout) we focus on looks like this:

![Plate layout for experiments](https://github.com/tansey/functionalmf/raw/master/img/dose-experiment-design.png)

The actual code is flexible to the number of replicates and doses. It does, however, make certain assumptions about the experiment:

1) The model assumes the scientist has filled the plate column-wise using a multi-headed pipette. Thus, every column (i.e. every concentration level) has a biased mean based on how many cells were drawn for that column.

2) Given a specific column (dose), the model assumes noise around that mean in terms of the initial population is i.i.d.

3) For each drug, it is assumed that the model can only have negative effects on the cells. Each drug is therefore bounded to the [0,1] range in terms of effect on the population size.

## Empirical Bayes likelihood
The likelihood for the outcomes is estimated by an empirical Bayes procedure. The model attempts to estimate the level of technical error. It assumes that if the lowest concentration has a mean higher than the control mean, the lowest value had no effect. It then estimates the distribution of initial column means using the distribution of these no-effect lowest doses. The estimation procedure uses an Efron-style Poisson GLM histogram fit to approximate the prior and looks something like this:

![Example empirical Bayes fit for population prior mean](https://github.com/tansey/functionalmf/raw/master/img/dose-empirical-bayes.png)

See the code in `empirical_bayes.py` for full details.

## Running the dose-response model
To fit your code to data, you need to create a CSV file that looks like the following:
```csv
cell line,drug,concentration,outcome
Tumor0,Drug0,,1.0226432741712916
Tumor0,Drug0,,1.0541479864861412
Tumor0,Drug0,,1.0260092908296126
Tumor0,Drug0,,1.0399213709427866
Tumor0,Drug0,,1.0766716865903976
Tumor0,Drug0,,1.0449048610909626
Tumor0,Drug0,-9.12,1.093482391658113
Tumor0,Drug0,-9.12,1.0976531387553998
Tumor0,Drug0,-9.12,1.142974173159974
Tumor0,Drug0,-9.12,1.0799755083343539
Tumor0,Drug0,-9.12,1.1032005762515136
Tumor0,Drug0,-9.12,1.128211261900736
Tumor0,Drug0,-8.64,0.9406109315522669
Tumor0,Drug0,-8.64,0.9738587418557895
Tumor0,Drug0,-8.64,0.9083942263498824
Tumor0,Drug0,-8.64,0.9367586188006692
``` 
In the above:
- `Tumor0` is the ID of the tumor cell line.
- `Drug0` is the ID of the drug.
- `-9.12` and `-8.64` are the log-concentrations of the drug. An empty value for this means the line is for a control.
- `1.0226432741712916` and the other values in the `outcome` column are the survival measurements. These can be normalized or raw.

Once you have your CSV file, you just need to run `fit.py`:
```python fit.py --data my_data.csv```
See the `fit.py` file for complete information

## Simulating an experiment
If you want to simulate some fake data for testing or to understand the model, you can run `sim.py`. See the file for full details on command line arguments.











