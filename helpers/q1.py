import numpy as np
import matplotlib.pyplot as plt

from datetime import date, timedelta

def generate_cumulative_accidents(data: list):
    """
    Generates an array of cumulative accident counts over a period of 40,550 days.

    Each value in the returned array represents the total number of accidents that
    have occurred up to and including that day (index). The count remains constant
    between events and increments by 1 on the day of each new accident.

    Args
    ---
    data: iterable
      Sequence of time intervals (in days) between consecutive
        accident events. Each value represents the number of days
        elapsed since the previous accident.

    Returns
    ---
    cumulative_accidents: np.ndarray
        Array of length 40,550 containing the cumulative accident count
        per day. The first accident is recorded at index 0 (day 1).
    mean_rate: float
        The average rate of accidents per day over the entire period, calculated as
        the total number of accidents divided by 40,550 days.
    years: int
        The year corresponding to the last day in the data, calculated based on the
        total number of days and starting from the year 1851.

    """
    cumulative_accidents = np.zeros(40550) 
    cumulative_accidents[0] = 1 # first accident occurs at time 0 (day 1)
    i = 0 
    j = 0 # tracks the index until which to fill the current accident count
    for interval in data:
        start = cumulative_accidents[i]
        j += int(interval)
        while i < j:
            cumulative_accidents[i] = start
            i += 1 # This ends when i == j
        cumulative_accidents[j] = start + 1
        mean_rate = cumulative_accidents[-1] / 40550

    return cumulative_accidents, mean_rate

def plot_cumulative_accidents(cumulative_accidents: list, mean_rate: float, savefig: str = "plots/cumulative_accidents.png"):
    """
    Plot the cumulative number of accidents over time with the mean rate
    overlaid as a dashed line.

    Parameters
    ---
    cumulative_accidents : np.ndarray
        Cumulative count of accidents at each time step, shape (40550,).
    mean_rate : float
        Mean rate of accidents per time step, used to plot the expected
        linear trend.
 
    savefig : str or None, optional
        File path to save the figure. If None, the figure is not saved.
    """
    start_date = date(1851, 3, 15)
    tick_years = range(1850, 1970, 10)  # every 10 years
    tick_positions = [(date(y, 1, 1) - start_date).days for y in tick_years]
    tick_labels = [str(y) for y in tick_years]
    
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(range(40550), cumulative_accidents, label="Cumulative Accidents")
    plt.plot(range(40550), mean_rate * np.arange(40550), linestyle="--", label=f"Mean Rate {mean_rate:.5f} per day = {mean_rate*365:.2f} per year")
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.title("Cumulative Number of Accidents Over Time")
    plt.xlabel("Time (Year)")
    plt.ylabel("Number of accidents")
    plt.legend()
    if savefig:
        plt.savefig(savefig)
    plt.show()