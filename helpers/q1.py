import numpy as np
import matplotlib.pyplot as plt

def generate_cumulative_accidents(data):
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

def plot_cumulative_accidents(cumulative_accidents, mean_rate):
    plt.figure(figsize=(7, 4))
    plt.plot(range(40550), cumulative_accidents, label="Cumulative Accidents")
    plt.plot(range(40550), mean_rate * np.arange(40550), linestyle="--", label="Mean Rate")
    plt.title("Cumulative Number of Accidents Over Time")
    plt.xlabel("Time")
    plt.ylabel("Number of accidents")
    plt.legend()
    plt.show()