import numpy as np
from sanpy.util.find_outliers import modified_zscore


def compute_apparent_velocity(data, time, distances):
    ntr = data.shape[0]
    arrival_time = []

    for i in range(0, ntr):
        t = np.argmax(data[i, :] ** 2)
        arrival_time.append(time[t])

    arrival_time = np.array(arrival_time)
    _, inliers = modified_zscore(arrival_time)
    c1, c2 = np.polyfit(distances[inliers], arrival_time[inliers], deg=1)

    return c1, c2
