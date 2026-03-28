import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from collections import Counter
from tqdm import tqdm


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

def plot_rate(rjmcmc, tau):
    model_chain = rjmcmc.chain
    thinned_chain = model_chain[int(10*tau)::int(2*tau)] # discard 10tau and thin by 2tau
    print(f"Number of samples after discarding and thinning: {len(thinned_chain)}")

    height_v_times = [] # heights at different times

    for i in tqdm(range(0, len(thinned_chain))):
        model = thinned_chain[i]
        k = (len(model) - 1) // 2

        # k change points and k+1 heigths
        change_points =  model[:k] # no plus 1 bc starts at 0 so exclusive of k is still k change points
        heights = model[k:]
        height_v_time = np.zeros(40550)
        current = 0
        next = 0
        edges = np.concatenate([[0], change_points.astype(int), [40550]])
        for j in range(0, len(edges) -1):
            #  height between these two edges
            height = heights[j]
            delta = edges[j+1] - edges[j]
            next = current + delta
            height_v_time[current:next] = height
            current = next
        height_v_times.append(height_v_time)
            

    # plot the mean hight at each time
    mean_height_v_time = np.mean(height_v_times, axis=0)
    low_50, high_50 = np.percentile(height_v_times, [25,75], axis = 0)
    low_90, high_90 = np.percentile(height_v_times, [5,95], axis = 0)

    t = range(40550)
    plt.figure(figsize=(20, 8))
    plt.fill_between(t, low_90, high_90, alpha=0.2, color='blue', label=r'90% uncertainty')
    plt.fill_between(t, low_50, high_50, alpha=0.4, color='blue', label=r'50% uncertainty')
    plt.plot(t, mean_height_v_time, color = 'red', label = "Inferred mean")
    plt.xlabel("Time (days)")
    plt.ylabel("Accident rate")
    plt.title("Inferred Accident Rate")
    plt.legend()
    plt.show()      
            