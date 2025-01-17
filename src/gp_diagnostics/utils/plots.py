
from scipy.stats import norm
import numpy as np

def snorm_qq(x):
    """
    Function for calculating standard normal QQ plot data with 95% confidence. Based on qqnorm in R.
    Input:
    x - data in 1D array
    Output:
    q_sample - sample quantiles
    q_snorm - standard normal quantiles
    q_snorm_upper - 95% upper band
    q_snorm_lower - 95% lower band
    """

    n = len(x) # Number of data points

    # Sample quantiles
    q_sample = np.sort(x)

    # Cumulative probabilities used to extract quantiles
    p = (np.arange(n) + 0.5) / n

    # Theoretical quantiles
    q_snorm = norm.ppf(p)

    # Confidence intervals are calculated using +/- k, where
    k = 0.895 / (np.sqrt(n) * (1- 0.01 / np.sqrt(n) + 0.85/n))

    q_snorm_upper = norm.ppf(p + k)
    q_snorm_lower = norm.ppf(p - k)

    return q_sample, q_snorm, q_snorm_upper, q_snorm_lower