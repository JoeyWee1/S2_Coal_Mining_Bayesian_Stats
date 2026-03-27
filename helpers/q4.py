import numpy as np
import emcee
import matplotlib.pyplot as plt
import dynesty
import corner
from scipy.stats import gaussian_kde, beta, gamma
from scipy.special import gammaln, betaln

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
    """
    Plot trace plots for all 2k+1 parameters of the k change-point model.

    Displays the first 500 thinned steps (or all steps if fewer than 500
    are available) for each walker, overlaid in black with transparency.
    All walkers are plotted on the same axis to visually assess mixing and
    convergence — a well-mixed chain shows walkers overlapping throughout
    with no visible drift or separation.

    Parameters
    ----------
    k : int, optional
        Number of change points, giving ndim = 2k+1 parameters. Default 1.
    samples : np.ndarray of shape (nsteps, nwalkers, ndim)
        Thinned posterior samples with walkers kept separate, as returned
        by generate_chain with flat=False.
    """
    print(f"There are {k} change points")
    ndim = 2*k + 1
    fig, ax = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    labels =  [f"Change point $s_{i+1}$ (days)" for i in range(k)] + [f"Height $h_{j}$" for j in range(k+1)]

    for i in range(ndim):
        ax[i].plot(samples[:500,  :, i], 'k', alpha=0.3)
        ax[i].set_ylabel(labels[i])

    ax[-1].set_xlabel("step")
    plt.show()

def corner_plot(k=1, samples=None):
    """
    Plot a corner plot of the posterior samples for the k change-point model.

    Displays the 1D marginal posteriors on the diagonal with vertical lines
    at the posterior mean (solid blue) and mean ± 1 standard deviation
    (dashed blue). Off-diagonal panels show 2D marginal posteriors as
    density contours. Titles show the median and 16th/84th percentile
    credible intervals as computed by corner by default.

    Parameters
    ----------
    k : int, optional
        Number of change points, giving ndim = 2k+1 parameters. Default 1.
    samples : np.ndarray of shape (nsamples, ndim)
        Flattened posterior samples with burn-in discarded, as returned
        by sampler.get_chain(flat=True, discard=burn_in).
    """
    labels = [f"Change point $s_{i+1}$ (days)" for i in range(k)] + [f"Height $h_{j}$" for j in range(k+1)]

    means = np.mean(samples, axis=0)
    stds = np.std(samples, axis=0)

    # For each parameter, pass [mean-std, mean, mean+std] as the truths-equivalent lines
    quantiles_per_param = [
        [means[i] - stds[i], means[i], means[i] + stds[i]]
        for i in range(2*k + 1)
    ]

    fig = corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        title_fmt=".4f",
        truths=means,           # plots vertical line at mean
        truth_color="blue",
    )
    axes = np.array(fig.axes).reshape((2*k+1, 2*k+1))

    for i in range(2*k + 1):
        ax = axes[i, i]
        ax.axvline(means[i], color="blue", label="mean")
        ax.axvline(means[i] - stds[i], color="blue", linestyle="--", label=r"$\pm 1\sigma$")
        ax.axvline(means[i] + stds[i], color="blue", linestyle="--")

    plt.show()


def gr_stat(n_chains=10, cumulative_days=None):
    """
    Compute the Gelman-Rubin R-hat diagnostic for the k=1 change-point model
    by running n_chains independent MCMC chains and comparing their variance.

    Each chain is run independently with generate_chain using a fixed random
    seed per chain, thinned by 2*tau to reduce autocorrelation, and flattened
    before comparison. Chains are truncated to the shortest chain length before
    computing R-hat.

    R-hat values close to 1.0 indicate convergence across all chains. Values
    above 1.1 suggest insufficient mixing and more steps are required.

    Parameters
    ----------
    n_chains : int, optional
        Number of independent chains to run. Default 10.
    cumulative_days : array_like
        Cumulative days on which accidents occurred, passed to generate_chain
        and LnPost as the data argument. Must be provided.

    Returns
    -------
    np.ndarray of shape (ndim,)
        R-hat statistic for each of the 2k+1 parameters. Values close to
        1.0 indicate convergence; values above 1.1 indicate poor mixing.
    """
    chains = []
    for i in range(n_chains): # we want to thin and flatten 
        sampler, mean_frac, taus, mean_tau, tau = generate_chain(steps=10000, cumulative_days=cumulative_days) 
        chains.append(sampler.get_chain(flat=True, thin = 2*tau)) # get the samples from the output tuple of generate_chain

    # Truncate to minimum length
    min_length = min(chain.shape[0] for chain in chains)
    chains = np.array([chain[:min_length] for chain in chains])  # (m, n, p)

    m, n, p = chains.shape

    chain_means = chains.mean(axis=1)           # (m, p)
    grand_mean  = chain_means.mean(axis=0)      # (p,)

    B = n / (m - 1) * np.sum((chain_means - grand_mean)**2, axis=0)  # (p,)
    W = chains.var(axis=1, ddof=1).mean(axis=0)                       # (p,)

    V_hat = (n - 1) / n * W + B / n
    return np.sqrt(V_hat / W)                   # (p,)

def gewecke(chain, first_frac=0.1, last_frac=0.5):
    """
    Compute the Geweke convergence diagnostic for a single MCMC chain.

    Compares the means of the first and last portions of the chain using a
    z-score. If the chain has converged, the two segments should be drawing
    from the same distribution and the statistic should be approximately
    standard normal, i.e. |z| < 2 for roughly 95% of converged parameters.

    Parameters
    ----------
    chain : array_like of shape (nsteps,)
        A 1D array of samples for a single parameter from a single chain,
        after thinning but before or after burn-in discard.
    first_frac : float, optional
        Fraction of the chain to use as the first segment. Default 0.1.
    last_frac : float, optional
        Fraction of the chain to use as the last segment. Default 0.5.

    Returns
    -------
    float
        The Geweke z-score. Values with |z| < 2 are consistent with
        convergence; |z| > 2 suggests the chain has not yet converged.
    """
    n = len(chain)
    na = int(n * first_frac)
    nb = int(n * last_frac)
    first_part = chain[:na]
    last_part = chain[-nb:]
    mean_first = np.mean(first_part)
    mean_last = np.mean(last_part)
    var_first = np.var(first_part, ddof=1)
    var_last = np.var(last_part, ddof=1)
    gewecke_stat = (mean_first - mean_last) / np.sqrt(var_first/na + var_last/nb)
    return gewecke_stat

def savage_dickey(samples):
    """
    Plot three diagnostic panels for the Savage-Dickey density ratio analysis
    of the k=1 change-point model M_1 against the constant rate model M_0.

    The Savage-Dickey ratio estimates the Bayes factor Z_0/Z_1 as the ratio
    of the posterior to prior density of delta = h1 - h0 at delta = 0.
    The difficulty is that the posterior is concentrated far from zero,
    making the KDE estimate at delta=0 unreliable with finite samples.

    Panels
    ------
    1 : 2D histogram of h0 vs h1 with the h1=h0 line overlaid. Points below
        the diagonal indicate h1 < h0, i.e. the rate decreased after s1.
    2 : Marginal posterior of the change point s1 over the full observation
        period [0, L=40550] days.
    3 : Posterior of delta = h1 - h0 with KDE overlay and prior density at
        delta=0 marked, illustrating the difficulty of the Savage-Dickey
        approach when the posterior has negligible support at delta=0.

    Parameters
    ----------
    samples : np.ndarray of shape (nsamples, 3)
        Flattened posterior samples with columns [s1, h0, h1], with
        burn-in discarded.
    """
    delta = samples[:, 2] - samples[:, 1]

    # KDE on delta
    kde = gaussian_kde(delta)
    delta_range = np.linspace(delta.min(), delta.max(), 1000)
    kde_vals = kde(delta_range)
    posterior_at_zero = kde(0)[0]

    # KDE on s1 for the second panel
    kde_s1 = gaussian_kde(samples[:, 0])
    s1_range = np.linspace(0, 40550, 1000)
    kde_s1_vals = kde_s1(s1_range)
    s1_posterior_at_zero = kde_s1(0)[0]
    s1_posterior_at_L = kde_s1(40550)[0]

    # The prior on s1
    x = np.linspace(0, 40550, 40550)
    L = 40550
    alpha = np.ones(2) * 2
    log_B = betaln(*alpha)
    B = np.exp(log_B)
    pi = 1 / ((L ** 3) *  B) * (x * (L-x))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: h0 vs h1 2D histogram
    h = ax[0].hist2d(samples[:, 1], samples[:, 2],
                     bins=100,
                     cmap='Blues',
                     density=False,
                     range=[[0, 0.015], [0, 0.015]])
    


    plt.colorbar(h[3], ax=ax[0], label='Counts')
    ax[0].plot([0, 0.015], [0, 0.015], 'r--', label='$h_1 = h_0$ manifold', zorder=5)
    ax[0].set_xlabel('$h_0$')
    ax[0].set_ylabel('$h_1$')
    ax[0].set_xlim(0, 0.015)
    ax[0].set_ylim(0, 0.015)
    ax[0].legend()
    ax[0].set_title('Posterior samples in rate space')

    # Panel 2: s1 marginal posterior
    ax[1].hist(samples[:, 0], bins=50, density=True)
    ax[1].set_xlabel('Change point $s_1$ (days)')
    ax[1].set_title('Marginal of change point $s_1$')
    ax[1].axvline(0, color='red', linestyle='--', label='$M_0 = M_1$')
    ax[1].axvline(40550, color='red', linestyle='--')
    ax[1].set_xlim(-10, 40600)
    ax[1].plot(s1_range, kde_s1_vals, 'b-', label='Posterior KDE')
    ax[1].scatter([0, 40550], [s1_posterior_at_zero, s1_posterior_at_L], color='black', zorder=5,
                  label=f'Posterior KDE at 0 and L = {s1_posterior_at_zero:.1f}, {s1_posterior_at_L:.1f}', marker='x')
    ax[1].plot(x, pi, label='Prior PDF of $s_1$', color='orange')
    ax[1].legend()

    # Panel 3: delta posterior with KDE
    ax[2].hist(delta, bins=50, density=True, alpha=0.4, label='Histogram')
    ax[2].plot(delta_range, kde_vals, 'b-', label='Posterior KDE')
    ax[2].axvline(0, color='red', linestyle='--', label='$\\delta = 0$')
    ax[2].scatter([0], [posterior_at_zero], color='black', zorder=5,
                  label=f'Posterior KDE at 0 = {posterior_at_zero:.1f}', marker='x')
    ax[2].set_xlabel('$\\delta = h_1 - h_0$')
    ax[2].set_title('Density estimate near the null subspace')
    ax[2].legend()

    plt.tight_layout()
    plt.show()

#----------------------------------------
# Nested Sampling helper functions
#----------------------------------------

def dynesty_ln_Likelihood(theta, data=None):
    """
    Compute the log-likelihood for the inhomogeneous Poisson process under
    model M_k, formatted for use with the dynesty nested sampling package.

    Identical in calculation to LnPost but returns only the log-likelihood
    without the log-prior contribution, as dynesty handles the prior
    separately via the prior transform function.

    Parameters
    ----------
    theta : array_like of shape (2k+1,)
        Model parameters: first k entries are the gaps between consecutive
        change points, followed by k+1 heights h_0, h_1, ..., h_k. The
        number of change points k is inferred from len(theta) as
        k = (len(theta) - 1) // 2.
    data : array_like, optional
        Cumulative days on which accidents occurred, i.e. the absolute
        times y_i of each of the N accidents within [0, L=40550].

    Returns
    -------
    float
        The log-likelihood ln L(data | theta, M_k), or -inf if any gap
        or height is non-positive.
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
    
    # print(f"theta: {len(theta)}, counts: {len(counts)}, seg_lengths: {len(seg_lengths)}, heights: {len(heights)}")
    LnL = np.sum(counts * np.log(heights) - heights * seg_lengths)
    return LnL

def dynesty_prior_transform(u):
    """
    Transform unit hypercube samples to the prior for the k=1 change-point
    model M_1, as required by dynesty's nested sampling interface.

    Dynesty samples uniformly from the unit hypercube u in [0,1]^ndim and
    passes each point to this function to map it to the physical parameter
    space. The transform must be the inverse CDF (percent point function)
    of each parameter's marginal prior.

    The parameter vector is [gap, h0, h1] where:
        - gap ~ Beta(2, 2) scaled to [0, L=40550]: the even-numbered order
          statistics prior for k=1 reduces to a single Beta(2,2) on [0,L].
        - h0, h1 ~ Gamma(alpha=2, beta=200): the height prior.

    Parameters
    ----------
    u : array_like of shape (3,)
        Unit hypercube sample with u[0] for the gap and u[1:] for the
        heights, each drawn uniformly from [0, 1] by dynesty.

    Returns
    -------
    np.ndarray of shape (3,)
        Physical parameters [gap, h0, h1] transformed from the unit
        hypercube according to the prior distributions.
    """
    gap_prior = [40550 * beta.ppf(u[0], 2, 2)]
    heights_prior = gamma.ppf(u[1:], a=2, scale=1/200)
    return np.concatenate([gap_prior, heights_prior])