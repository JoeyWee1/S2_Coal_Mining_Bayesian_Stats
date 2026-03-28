import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from collections import Counter


def autocorrelation_time(rjmcmc):
    """
    Estimate the integrated autocorrelation time tau for a 1D chain.
    
    Parameters
    ----------
    rjmcmc : RJMCMC
        A fitted RJMCMC object with a populated ``chain`` attribute.
    
    Returns
    -------
    tau : float
        Integrated autocorrelation time.
    """
    ks = [(len(x) - 1) // 2 for x in rjmcmc.chain]
    x = np.array(ks, dtype=float)
    n = len(x)
    x = x - x.mean()
    
    # normalised autocorrelation function via FFT
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= acf[0]
    
    # sum autocorrelation with automatic windowing (Sokal's method)
    tau = 0.5 # half plus sum terms
    C = 5.0
    for t in range(1, n): # This was summing to 0. Googled and learnt "sokal" windowing
        tau += acf[t] # Sokal sounds like one of those evil aliens from stargate lol
        if t >= C * tau: # https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
            break #  At longer lags,  starts to contain more noise than signal and summing all the way out to  will result in a very noisy estimate of tau
    
    return tau


def k_post_trace(rjmcmc, discard):
    r"""
    Plot the trace of the model index k and its posterior distribution.

    Displays a two-panel figure: the left panel shows the trace of k over
    MCMC iterations, and the right panel shows the posterior distribution
    of k as a histogram. The MAP estimate of k is indicated by a red
    dashed line.

    Parameters
    ----------
    rjmcmc : RJMCMC
        A fitted RJMCMC object with a populated ``chain`` attribute.
    discard: int
        Number of samples to remove at the start

    Returns
    -------
    k_map : int
        The MAP estimate of the number of change points.
    """
    ks = [(len(x) - 1) // 2 for x in rjmcmc.chain]
    counts = Counter(ks)
    k_map = counts.most_common(1)[0][0]

    discard = int(discard)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [2.5, 1]}, figsize=(10, 4))

    ax1.plot(ks, 'k')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Model with k change points')
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_ylim(0, 30)
    ax1.axvline(discard, linestyle = "--", color='red', label = f"Burn-in discard:{discard:1d}")
    # ax1.grid(which='both', axis='y', linestyle='--', linewidth=0.5)
    ax1.set_title('Trace plot of the number of change points k')
    ax1.legend()

    ax2.hist(ks[discard:], bins=np.arange(-0.5, 31.5, 1), orientation='horizontal', density=True, color='white', edgecolor='black')
    ax2.set_xlabel('Posterior probability')
    ax2.set_title('Posterior over k')
    ax2.axhline(k_map, color='red', linestyle='--', label=f'MAP k={k_map}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return k_map
