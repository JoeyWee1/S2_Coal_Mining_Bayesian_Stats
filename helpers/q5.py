import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from collections import Counter


def k_post_trace(rjmcmc):
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

    Returns
    -------
    k_map : int
        The MAP estimate of the number of change points.
    """
    ks = [(len(x) - 1) // 2 for x in rjmcmc.chain]
    counts = Counter(ks)
    k_map = counts.most_common(1)[0][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [2.5, 1]}, figsize=(10, 4))

    ax1.plot(ks, 'k')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Model with k change points')
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_ylim(0, 30)
    ax1.grid(which='both', axis='y', linestyle='--', linewidth=0.5)
    ax1.set_title('Trace plot of the number of change points k')

    ax2.hist(ks, bins=np.arange(-0.5, 31.5, 1), orientation='horizontal', density=True, color='white', edgecolor='black')
    ax2.set_xlabel('Posterior probability')
    ax2.set_title('Posterior over k')
    ax2.axhline(k_map, color='red', linestyle='--', label=f'MAP k={k_map}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return k_map
