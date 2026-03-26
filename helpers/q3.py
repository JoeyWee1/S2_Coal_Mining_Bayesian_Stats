import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

def find_z0():
    """
    Calculate the log-evidence for the constant rate model M0 by direct
    numerical integration over the single parameter h0.

    The integral is computed using the log-sum-exp trick to avoid numerical
    underflow: the log-posterior is shifted by its maximum value before
    exponentiation, then the shift is added back in log-space afterward.

    Returns
    -------
    float
        The log-evidence ln(Z0) = ln(P({I_i} | M0)).
    """
    h0_vals = np.linspace(1e-5, 0.02, 20000, dtype=np.longdouble)
    ln_post_vals = m0_ln_post(h0_vals)
    m = max(ln_post_vals) # Rescale for numerical stability
    ln_post_vals_shift = ln_post_vals - m
    unscaled_z = np.trapezoid(np.exp(ln_post_vals_shift), h0_vals)
    m0_lnz = m + np.log(unscaled_z)
    return m0_lnz

def plot_prior_post(m0_lnz):
    """
    Plot the prior and normalised posterior for the constant rate model M0.

    Computes the posterior mean and mode analytically from the conjugate
    Gamma posterior Γ(α + N, β + L), and overlays them as vertical lines.
    Uses the analytic Gamma posterior to plot 95 percent confidence interval.
    The posterior is normalised by subtracting the log-evidence in log-space
    before exponentiation to avoid numerical underflow.

    Parameters
    ----------
    m0_lnz : float
        The log-evidence ln(Z0) = ln(P({I_i} | M0)), used to normalise
        the posterior to a proper PDF for comparison with the prior.
    """
    h0_vals = np.linspace(1e-5, 0.02, 1000, dtype=np.longdouble)
    ln_post_vals = m0_ln_post(h0_vals)
    ln_pi_vals = m0_ln_pi(h0_vals)

    # The prior is a normalised PDF so to compare we need to normalise the posterior by subtracting its log-evidence
    ln_post_vals -= m0_lnz
    post_vals = np.exp(ln_post_vals)
    pi_vals = np.exp(ln_pi_vals)

    # Find the mean analytically
    alpha = 1
    beta = 200
    alpha_prime = alpha + 191
    beta_prime = beta + 40550
    post_mean = alpha_prime/beta_prime

    # Analytically evaluate 95% confidence interval
    lower = stats.gamma.ppf(0.025, a = alpha_prime, scale = 1/beta_prime)
    upper = stats.gamma.ppf(0.975, a = alpha_prime, scale = 1/beta_prime)


    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(h0_vals, pi_vals,
            label=rf"Prior $\pi(h_0|M_0)$ = $\Gamma(\alpha = 1, \beta= 200$")
    ax.plot(h0_vals, post_vals,
            label=rf"Posterior $p(h_0|d, M_0)$ = $\Gamma(\alpha={alpha_prime}, \beta={beta_prime})$")
    ax.axvline(post_mean, color="red", linestyle="-",
               label=f"Posterior mean = {post_mean:.6f}")
    ax.axvspan(lower, upper, alpha=0.2, color='orange', label="95% credible interval")
    ax.set_xlabel("h_0")
    ax.set_title("Constant Rate Prior vs Posterior")
    ax.legend()
    fig.show()







