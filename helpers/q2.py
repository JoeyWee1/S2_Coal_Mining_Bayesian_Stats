import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed=1701):
    """
    Sets the seed for NumPy's random number generator.

    Args:
        seed (int): The seed value to use for the random number generator.
    """
    np.random.seed(seed)

def find_gaps(samples, L=40550):
    """
    Compute the gaps between sorted sample points within a bounded interval.

    Prepends 0 and appends L to the sample array, then calculates the
    difference between each consecutive pair of points.

    Parameters
    ----------
    samples : array-like
        Sorted sample positions within the interval [0, L].
    L : int or float, optional
        Upper bound of the interval. Default is 40550.

    Returns
    -------
    list of float
        List of gap sizes between consecutive points (including boundaries).
        Length is len(samples) + 1.
    """
    left = [0]
    right = [L]
    full = np.concatenate([left, samples, right])
    gaps = [(full[i] - full[i-1]) for i in range(1, len(full))]
    return gaps

def even_stats(k=4, L=40550):
    """
    Generate k evenly-spaced-in-expectation random samples scaled to [0, L].

    Draws 2k+1 uniform random values, sorts them, then selects every other
    value starting from index 1 (the odd-indexed elements). This produces k
    samples with uniform order-statistic spacing. The result is scaled to [0, L].

    Parameters
    ----------
    k : int, optional
        Number of sample points to return. Default is 4.
    L : int or float, optional
        Upper bound of the scaling interval. Default is 40550.

    Returns
    -------
    numpy.ndarray
        Array of k sample values in ascending order, scaled to [0, L].
    """
    even_rand = [np.random.uniform() for i in range((2*k+1))]
    even_sorted = np.sort(even_rand)
    even_unscaled = even_sorted[1::2]    
    even = even_unscaled * L
    return even

def plain_stats(k=4, L=40550):
    """
    Generate k uniformly random samples scaled to [0, L].

    Draws k uniform random values, sorts them, and scales to [0, L].
    This is a standard uniform order-statistic sample with no spacing constraints.

    Parameters
    ----------
    k : int, optional
        Number of sample points to return. Default is 4.
    L : int or float, optional
        Upper bound of the scaling interval. Default is 40550.

    Returns
    -------
    numpy.ndarray
        Array of k sample values in ascending order, scaled to [0, L].
    """
    plain_rand = [np.random.uniform() for i in range(k)]
    plain_unscaled = np.sort(plain_rand) 
    plain = plain_unscaled * L
    return plain

def compare_gaps(k=4, samples = 100000):
    """
    Compare even and plain sampling strategies by visualising sample positions and gap distributions.

    Generates two sets of figures:
      1. A 4-panel stacked plot showing the distribution of each sample point's
         position (s1–s4) for even vs plain sampling, with a shared x-axis.
      2. A 2-panel plot showing the flattened gap size distributions for
         even vs plain sampling.

    Parameters
    ----------
    k : int, optional
        Number of sample points per draw. Default is 4.
    samples : int, optional
        Number of Monte Carlo draws used to build the distributions. Default is 10000.

    Returns
    -------
    None
        Displays plots inline; does not return any values.
    """
    set_seed()
    even_gaps = []
    even_points = []
    plain_gaps = []
    plain_points = []

    for i in range(0, samples):
        even_point = even_stats(k=k)
        plain_point = plain_stats(k=k)
        even_points.append(even_point)
        plain_points.append(plain_point)
        even_gaps.append(find_gaps(even_point))
        plain_gaps.append(find_gaps(plain_point)) 

    even_gaps = np.array(even_gaps).flatten()
    plain_gaps = np.array(plain_gaps).flatten()
    even_points = np.array(even_points)
    plain_points = np.array(plain_points)

    # Plot the positions of the change points
    fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True)

    for i, ax in enumerate(axes):
        ax.hist(even_points[:,i], bins=75, alpha=0.5, label='Even')
        ax.hist(plain_points[:,i], bins=75, alpha=0.5, label='Plain')
        ax.legend()
        ax.set_ylabel('Frequency')
    axes[3].set_xlabel('Position')

   
    plt.tight_layout()
    fig.show()

    # Plot the distributions of the gaps
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax[0].hist(even_gaps, bins=75)
    ax[0].set_title("Even-stats gaps")
    ax[0].set_ylabel('Frequency')
    ax[0].axvline(np.mean(even_gaps), color='red', linestyle='--', label=f'Mean: {np.mean(even_gaps):.1f}')
    ax[0].axvline(np.percentile(even_gaps, 5),  color='orange', linestyle=':', label=f'5th pct: {np.percentile(even_gaps, 5):.1f}')
    ax[0].axvline(np.percentile(even_gaps, 95), color='orange', linestyle=':', label=f'95th pct: {np.percentile(even_gaps, 95):.1f}')
    ax[0].legend()

    ax[1].hist(plain_gaps, bins=75)
    ax[1].set_title("Plain-stats gaps")
    ax[1].set_xlabel('Gap size')
    ax[1].set_ylabel('Frequency')
    ax[1].axvline(np.mean(plain_gaps), color='red', linestyle='--', label=f'Mean: {np.mean(plain_gaps):.1f}')
    ax[1].axvline(np.percentile(plain_gaps, 5),  color='orange', linestyle=':', label=f'5th pct: {np.percentile(plain_gaps, 5):.1f}')
    ax[1].axvline(np.percentile(plain_gaps, 95), color='orange', linestyle=':', label=f'95th pct: {np.percentile(plain_gaps, 95):.1f}')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    even_5th = np.percentile(even_gaps, 5)
    proportion = np.mean(plain_gaps < even_5th)
    print(f'Proportion of plain gaps below 5th percentile of even gaps: {proportion:.3f}')

