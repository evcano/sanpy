import numpy as np


def compute_snr(tr, signal_window, noise_dur):
    noise_window = [signal_window[1] + 1,
                    signal_window[1] + 1 + int(noise_dur / tr.stats.delta)]

    if noise_window[1] >= tr.stats.npts:
        print('Cannot compute SNR: noise window is outside trace limits')
        snr = -999.0

    signal = tr.data[signal_window[0]:signal_window[1]+1].copy()
    noise = tr.data[noise_window[0]:noise_window[1]+1].copy()

    a = np.max(np.abs(signal))
    b = np.sqrt(np.mean(np.square(noise)))
    snr = a / b

    if snr < 0:
        snr = -10.0 * np.log10(-snr)
    elif snr > 0:
        snr = 10.0 * np.log10(snr)

    return snr
