import numpy as np
from scipy.special import gammaln

def m0_ln_pi(h0):
    """
    Compute the log prior density for the m0 model at h0.

    Parameters
    ----------
    h0 : float or array-like
        The value(s) at which to evaluate the prior.

    Returns
    -------
    float or numpy.ndarray
        Log prior density evaluated at h0.
    """
    return np.log(200) -200 * h0


def m0_ln_post(h0):
    """
    Compute the unnormalised log posterior for the M0 (constant rate) model.

    Combines the Gamma(1, 200) log prior with the Poisson process log likelihood,
    giving the log posterior up to a normalising constant.

    Log prior:      log(200) - 200 * h0
    Log likelihood: 191 * log(h0) - 40550 * h0
    Log posterior:  log(200) + 191 * log(h0) - 40750 * h0

    Parameters
    ----------
    h0 : float or array-like
        Constant accident rate (events per day).

    Returns
    -------
    float or numpy.ndarray
        Unnormalised log posterior evaluated at h0.
    """
    return np.log(200) + 191 * np.log(h0)  -40750 * h0

def find_z0





