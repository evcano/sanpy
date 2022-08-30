import matplotlib.mlab as mlab
import numpy as np
from scipy import interpolate
from scipy.signal import correlation_lags


def compute_single_psd(data, par):
    segnpts = 1024
    noverlap = segnpts // 2
    stations_psd = []

    for i in range(0, data.shape[0]):
        pxx, freqs = mlab.psd(data[i, :],
                              Fs=par['fs'],
                              NFFT=segnpts,
                              noverlap=noverlap,
                              detrend="mean")

        # normalize psd amplitude so it matches fft amplitude
        pxx = np.sqrt(par['fs'] * par['nfft'] * 0.5 * pxx)
        stations_psd.append(pxx)

    stations_psd = np.asarray(stations_psd)
    single_psd = np.percentile(stations_psd, q=95, axis=0)

    # interpolate so len(array_psd) = len(fft[:h_nfft+1])
    interp = interpolate.interp1d(freqs, single_psd)

    freq_fft = np.fft.rfftfreq(par['nfft'], par['dt'])
    single_psd2 = interp(freq_fft)

    return single_psd2


def my_centered(arr, newsize):
    '''
    Use to eliminate the zero-padding effect on correlations. Taken from NOISI
    '''

    if newsize % 2 == 0:
        raise ValueError('Newsize must be odd.')

    # pad with zeros, if newsize > len(arr)
    newarr = np.zeros(newsize)

    # get the center portion of a 1-dimensional array correctly
    n = len(arr)
    i0 = (n - newsize) // 2
    if i0 < 0:
        i0 = (newsize - n) // 2
        newarr[i0: i0 + n] += arr

    else:
        if n % 2 == 0:
            # This is arbitrary
            # because the array has no 'center' sample
            i0 += 1
        newarr[:] += arr[i0: i0 + newsize]

    return newarr


def remove_transient_signal(tr, transient_thresholds):
    max_energy = np.max(tr.data ** 2)

    sta = '{}.{}'.format(tr.stats.network, tr.stats.station)
    idx = transient_thresholds[0].index(sta)

    if max_energy > transient_thresholds[1][idx]:
        return True
    else:
        return False


def transient_signal_thresholds(st, corr_dur, corr_overlap, thr):
    stations = []
    stations_thr = []

    # loop over traces
    for tr in st:
        stations.append('{}.{}'.format(tr.stats.network,
                                       tr.stats.station))
        energies = []

        # sliding window over trace
        for win_tr in tr.slide(corr_dur, corr_overlap):
            max_energy = np.max(win_tr.data ** 2.0)
            energies.append(max_energy)

        energies = np.array(energies)
        stations_thr.append(thr * np.median(energies))

    return [stations, stations_thr]


def uniform_time_normalization(corr):
    max_amps = np.max(np.abs(corr), axis=1)
    a = np.percentile(max_amps, 95)
    corr = np.divide(corr, a)

    return corr


def uniform_spectral_whitening(fft, single_psd):
    fft = np.divide(fft, single_psd)

    return fft


def xcorr(fft, stations, pairs, par):
    corr = []

    lags = correlation_lags(par['corr_npts'], par['corr_npts'])

    maxlag = int(par["maxlag"] / par['dt'])  # maxlag to store (in samples)
    store_lags = np.where(np.abs(lags) <= maxlag)[0]

    for pair in pairs:
        sta1, sta2 = pair.split("_")
        idx1 = stations.index(sta1)
        idx2 = stations.index(sta2)

        # linear cross-correlation of sta1 with sta2 as in equation 11 of
        # Tromp et al. 2010
        tmp_corr = fft[idx1, :] * np.conj(fft[idx2, :])

        # convert to time domain, this results in [pos_lags, neg_lags]
        tmp_corr = np.real(np.fft.irfft(tmp_corr, par['nfft'],
                                        norm="backward"))

        # switch second and first halves of corr to obtain [neg_lags, pos_lags]
        tmp_corr = np.fft.fftshift(tmp_corr)

        # eliminate effect of zero-padding
        tmp_corr = my_centered(tmp_corr, len(lags))

        # store lags of interest
        tmp_corr = tmp_corr[store_lags]

        corr.append(tmp_corr)

    corr = np.array(corr)

    return corr
