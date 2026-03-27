from scipy.special import gammaln
import numpy as np
import emcee
import matplotlib.pyplot as plt

def LnPost(theta, data):
    """
    Compute the log-posterior for model M_k under the inhomogeneous Poisson
    process with k change points.

    The parameter vector theta encodes k gaps between change points followed
    by k+1 heights. The final gap to L=40550 is inferred implicitly so that
    the change points always span the full observation period. Returns -inf
    immediately if any gap or height is non-positive (prior boundary).

    The log-prior has two contributions:
        - Change points: even-numbered order statistics prior, proportional
          to the product of all k+1 gaps (including the final gap to L),
          normalised by the Dirichlet B(2,2,...,2) function.
        - Heights: independent Gamma(alpha=1, beta=200) priors, giving a
          contribution of (k+1)*log(200) - 200*sum(heights).

    The log-likelihood is that of an inhomogeneous Poisson process:
        ln L = sum_j [ n_j * log(h_j) - h_j * l_j ]
    where n_j is the number of accidents and l_j is the length of segment j.

    Parameters
    ----------
    theta : array_like of shape (2k+1,)
        Model parameters: first k entries are the gaps between consecutive
        change points (s_1, s_2-s_1, ..., s_k-s_{k-1}), followed by k+1
        heights h_0, h_1, ..., h_k. The number of change points k is
        inferred from len(theta) as k = (len(theta) - 1) // 2.
    data : array_like
        Cumulative days on which accidents occurred, i.e. the absolute
        times y_i of each of the N accidents within [0, L].

    Returns
    -------
    float
        The log-posterior ln P(theta | data, M_k), or -inf if theta lies
        outside the support of the prior.
    """
    k = int((len(theta) - 1) / 2)
    gaps = theta[0:k]
    heights = theta[k:]

    # final gap 
    final_gap = 40550 - np.sum(gaps)
    gaps = np.append(gaps, final_gap)

    # safety
    if np.any(gaps <= 0):
        return -np.inf
    if np.any(heights <= 0):
        return -np.inf

    # log prior calculation: chance points prior
    alpha = np.ones(k+1) * 2

    ln_gaps = np.log(gaps)
    sum_ln_gaps = np.sum(ln_gaps)

    lnB = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))

    change_portion = (-1 * (2*k + 1)*np.log(40550)) - lnB + sum_ln_gaps  # the contribution of the change points to the prior

    # log prior calculation: heights prior
    height_portion = ((k+1) * np.log(200)) - (200 * np.sum(heights))

    # LnPrior
    LnPi = height_portion + change_portion

    # reconstruct change points
    change_points = np.cumsum(gaps[:-1])
    edges = np.concatenate([[0], change_points, [40550]])
    seg_lengths = np.diff(edges)

    # count data in each segment
    counts = np.zeros(k + 1, dtype=int)
    for j in range(k + 1):
        left = edges[j]
        right = edges[j + 1]
        counts[j] = np.sum((data > left) & (data < right))

    LnL = np.sum(counts * np.log(heights) - heights * seg_lengths)

    # log posterior
    LnPost = LnPi + LnL

    return LnPost


def generate_chain(k = 1, nwalkers = 32, steps = 10000, tf=2, cumulative_days=None):
    """
    Run an emcee ensemble MCMC sampler for the k change-point model M_k.

    Initialises nwalkers walkers by drawing gaps from a symmetric Dirichlet
    distribution scaled to L=40550 and heights from the Gamma(1, 200) prior.
    The chain is thinned by tf * max(autocorrelation times) across all
    parameters to reduce sample correlation.

    Note: the returned samples have discard=0, meaning no burn-in steps are
    removed. The caller is responsible for inspecting the trace plot and
    discarding the initial equilibration period before using the samples for
    inference.

    Parameters
    ----------
    k : int, optional
        Number of change points, giving ndim = 2k+1 parameters. Default 1.
    nwalkers : int, optional
        Number of ensemble walkers. Must be at least 2*ndim. Default 32.
    steps : int, optional
        Number of MCMC steps per walker. Default 10000.
    tf : int, optional
        Thinning factor multiplier applied to the maximum autocorrelation
        time, i.e. samples are thinned by tf * max(taus). Default 2.
    cumulative_days : array_like
        Cumulative days on which accidents occurred, passed to LnPost
        as the data argument. Must be provided.

    Returns
    -------
    sampler : emcee.EnsembleSampler
        The completed sampler object, containing the full unthinned chain.
    mean_frac : float
        Mean acceptance fraction across all walkers. Healthy values are
        typically in the range 0.2 to 0.5.
    taus : np.ndarray of shape (ndim,)
        Estimated autocorrelation time for each parameter dimension.
    mean_tau : float
        Mean autocorrelation time across all parameters.
    tau : int
        Maximum autocorrelation time across all parameters, used as the
        base thinning interval.
    samples : np.ndarray of shape (nwalkers * steps / (tf * tau), ndim)
        Thinned and flattened posterior samples with no burn-in discarded.
    """

    rng = np.random.default_rng(1701)

    ndim = 2*k + 1 # k gaps (plus 1 implied) and k+1 heights
    
    # Initial positions for all walkers: shape (nwalkers, ndim)
    initial_positions = np.ones((nwalkers, ndim)) # shape (nwalkers, ndim)
    for w in range(nwalkers):
        # k positive gaps summing to L
        gaps = rng.dirichlet(np.ones(k+1) * 2.0) * 40550
        gaps = gaps[:-1] # drop the final gap since it's implied by the others

        # k+1 positive heights from the prior
        heights = rng.exponential(scale=1/200, size=k+1)

        initial_positions[w] = np.concatenate([gaps, heights])

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        LnPost,
        args=[cumulative_days],
)

    sampler.run_mcmc(initial_positions, steps, progress=True)

    mean_frac =sampler.acceptance_fraction.mean()

    taus = sampler.get_autocorr_time() # for thinning
    mean_tau = np.mean(taus)
    tau = int(max(taus)) # use the maximum autocorrelation time across all parameters for thinning

    print(f"Mean acceptance fraction: {mean_frac:.3f}")
    print(f"Mean autocorrelation time: {mean_tau:.2f} steps")

    return sampler, mean_frac, taus, mean_tau, tau

def trace_plot(k=1, samples=None):
    print(f"There are {k} change points")
    ndim = 2*k + 1
    fig, ax = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    labels = [f"Change point s{i+1}" for i in range(k)] + [f"Height h{j}" for j in range(k+1)]

    for i in range(ndim):
        ax[i].plot(samples[:500,  :, i], 'k', alpha=0.3)
        ax[i].set_ylabel(labels[i])

    ax[-1].set_xlabel("step")
    plt.show()