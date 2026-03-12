import numpy as np

data = np.array([1.570e+02, 6.500e+01])
cumulative_accidents = np.zeros(40550)
cumulative_accidents[0] = 1
i = 0 
j = 0
for interval in data:
    start = cumulative_accidents[i]
    j += interval
    while i < j:
        cumulative_accidents[i] = start
        i += 1
    cumulative_accidents[j] = start + 1

print(cumulative_accidents[40000:-1])
