import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy import stats

import warnings

warnings.filterwarnings("ignore")

from scipy.stats import (
    norm, beta, expon, gamma, genextreme, logistic, lognorm, triang, uniform, fatiguelife,
    gengamma, gennorm, dweibull, dgamma, gumbel_r, powernorm, rayleigh, weibull_max, weibull_min,
    laplace, alpha, genexpon, bradford, betaprime, burr, fisk, genpareto, hypsecant,
    halfnorm, halflogistic, invgauss, invgamma, levy, loglaplace, loggamma, maxwell,
    mielke, ncx2, ncf, nct, nakagami, pareto, lomax, powerlognorm, powerlaw, rice,
    semicircular, rice, invweibull, foldnorm, foldcauchy, cosine, exponpow,
    exponweib, wald, wrapcauchy, truncexpon, truncnorm, t, rdist
)

distributions = [
    norm, beta, expon, gamma, genextreme, logistic, lognorm, triang, uniform, fatiguelife,
    gengamma, gennorm, dweibull, dgamma, gumbel_r, powernorm, rayleigh, weibull_max, weibull_min,
    laplace, alpha, genexpon, bradford, betaprime, burr, fisk, genpareto, hypsecant,
    halfnorm, halflogistic, invgauss, invgamma, levy, loglaplace, loggamma, maxwell,
    mielke, ncx2, ncf, nct, nakagami, pareto, lomax, powerlognorm, powerlaw, rice,
    semicircular, rice, invweibull, foldnorm, foldcauchy, cosine, exponpow,
    exponweib, wald, wrapcauchy, truncexpon, truncnorm, t, rdist
]

ksN = 100  # Kolmogorov-Smirnov KS test for goodness of fit: samples
ALPHA = 0.05  # significance level for hypothesis test


# KS test for goodness of fit

def kstest(data, distname, paramtup):
    ks = stats.kstest(data, distname, paramtup, ksN)[1]  # return p-value
    return ks  # return p-value

# distribution fitter and call to KS test

def fitdist(data, dist):
    fitted = dist.fit(data, floc=0.0)
    ks = kstest(data, dist.name, fitted)
    res = (dist.name, ks, *fitted)
    return res

def histData (data):
    fig, ax = plt.subplots(1, 1)
    ax.hist(data, density=True, histtype='stepfilled', alpha=0.2)
    ax.legend(loc='best', frameon=False)
    plt.show()

def brute_force(data):
    # call fitting function for all distributions in list
    res = [fitdist(data, D) for D in distributions]

    # convert the fitted list of tuples to dataframe
    pd.options.display.float_format = '{:,.3f}'.format
    df = pd.DataFrame(res, columns=["distribution", "KS p-value", "param1", "param2", "param3", "param4", "param5"])
    df["distobj"] = distributions
    df.sort_values(by=["KS p-value"], inplace=True, ascending=False)
    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)
