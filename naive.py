# library imports
import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from timeit import timeit
from pandas.api.types import is_numeric_dtype
from scipy.stats._continuous_distns import _distn_names


class NaiveApproach:
    """
    The naive approach (bruteforce) fits all distributions using 'scipy' built-in function 'fit
    that maximaizes the MLE. The best distribution is picked according to minimal SSE value.
    """

    def __init__(self):
        pass

    @staticmethod
    def run(df):
        answers = []
        times = []
        for col in list(df):
            if is_numeric_dtype(df[col]):
                start = timeit()
                data = pd.Series(df[col])

                # Find best fit distribution
                best_distibutions = NaiveApproach.best_fit_distribution(data=data, bins=200)
                best_dist = best_distibutions[0]

                # Make PDF with best params
                # pdf = NaiveApproach.make_pdf(best_dist[0], best_dist[1])
                end = timeit()
                times.append(end - start)

                answers.append(best_dist[2])
        return answers, times

    @staticmethod
    def best_fit_distribution(data, bins=200):
        """Model data by finding best fit distribution to data"""
        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        y_raw = data.values
        x_raw = data.index.values

        # Best holders
        best_distributions = []

        # Estimate distribution parameters from data
        for ii, distribution in enumerate([d for d in _distn_names if d in ['norm', 'uniform']]):

            # print("{:>3} / {:<3}: {}".format(ii + 1, len(_distn_names), distribution))

            distribution = getattr(st, distribution)

            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]


                    sse = np.inf
                    # Calculate fitted PDF and error with fit in distribution
                    # If distribution is 'uniform' check against original values
                    if distribution.name == 'uniform':
                        theor_dist = [loc for i in range(len(y_raw))]
                        sse = np.sum(np.power(y_raw - theor_dist, 2.0))
                    else:
                        pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                        sse = np.sum(np.power(y - pdf, 2.0))

                    # identify if this distribution is better
                    best_distributions.append((distribution, params, sse))

            except Exception:
                pass

        return sorted(best_distributions, key=lambda x: x[2])

    @staticmethod
    def make_pdf(dist, params, size=10000):
        """Generate distributions's Probability Distribution Function """

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Get sane start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)

        return pdf
