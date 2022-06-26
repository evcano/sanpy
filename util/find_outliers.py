import numpy as np


def modified_zscore(data, thr=3.5):
    median = np.median(data)
    median_dev = np.abs(data - median)
    median_ad = np.median(median_dev)

    if median_ad == 0:
        mean = np.mean(data)
        mean_dev = np.abs(data - mean)
        mean_ad = np.mean(mean_dev)
        zscores = (data - median) / (1.253314 * mean_ad)
    else:
        zscores = (data - median) / (1.486 * median_ad)

    outliers = np.where((zscores >= thr) | (zscores <= -thr))[0]
    inliers = np.where((zscores < thr) & (zscores > -thr))[0]

    return outliers, inliers
