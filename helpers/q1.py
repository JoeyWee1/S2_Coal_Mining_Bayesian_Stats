import numpy as np

def generate_cumulative_accidents(data):
    """
    Generates an array of cumulative accident counts over a period of 40,550 days.

    Each value in the returned array represents the total number of accidents that
    have occurred up to and including that day (index). The count remains constant
    between events and increments by 1 on the day of each new accident.

    Args:
        data (iterable): Sequence of time intervals (in days) between consecutive
                         accident events. Each value represents the number of days
                         elapsed since the previous accident.

    Returns:
        np.ndarray: Array of length 40,550 containing the cumulative accident count
                    per day. The first accident is recorded at index 0 (day 1).

    Example:
        >>> generate_cumulative_accidents([3, 5, 2])
        # Day 0: 1 accident, days 1–2: 1, day 3: 2, days 4–7: 2, day 8: 3, ...
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
    return cumulative_accidents