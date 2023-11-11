import matplotlib.mlab as mlab
import numpy as np
from obspy.core import UTCDateTime
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

        pxx = np.sqrt(pxx)
        stations_psd.append(pxx)

    stations_psd = np.asarray(stations_psd)
    single_psd = np.percentile(stations_psd, q=95, axis=0)

    # interpolate so len(array_psd) = len(fft[:h_nfft+1])
    interp = interpolate.interp1d(freqs, single_psd)

    freq_fft = np.fft.rfftfreq(par['corr_nfft'], par['dt'])
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


def read_bad_windows(file_):
    win_dates = np.loadtxt(file_, dtype=str)
    dates_utc = {}
    for w in win_dates:
        utc = UTCDateTime(w)
        if utc.date.isoformat() not in dates_utc.keys():
            dates_utc[utc.date.isoformat()] = []
        dates_utc[utc.date.isoformat()].append(utc)
    return dates_utc


def uniform_time_normalization(corr):
    max_amps = np.max(np.abs(corr), axis=1)
    a = np.percentile(max_amps, 95)
    corr = np.divide(corr, a)

    return corr


def xcorr(data_fft, stations_win, corr_cmp, P):
    lags = correlation_lags(P.par['corr_npts'], P.par['corr_npts'])
    maxlag = int(P.par["maxlag"] / P.par['dt'])  # maxlag to store (in samples)
    store_lags = np.where(np.abs(lags) <= maxlag)[0]

    assert(corr_cmp in ["EE","NN","ZZ"]),("incorrect component")
    c = corr_cmp[0]

    pairs_win = []
    corr = []

    for pair in P.pairs_list:
        sta1, sta2 = pair.split("_")

        flag1 = sta1 in stations_win[c]
        flag2 = sta2 in stations_win[c]

        if not flag1 or not flag2:
            continue

        i1 = stations_win[c].index(sta1)
        i2 = stations_win[c].index(sta2)

        # linear cross-correlation of sta1 with sta2 as in equation 11 of
        # Tromp et al. 2010
        tmp_corr = data_fft[c][i1, :] * np.conj(data_fft[c][i2, :])

        # convert to time domain, this results in [pos_lags, neg_lags]
        tmp_corr = np.real(np.fft.irfft(tmp_corr, P.par['corr_nfft'],
                                        norm="backward"))

        # switch second and first halves of corr to obtain [neg_lags, pos_lags]
        tmp_corr = np.fft.fftshift(tmp_corr)

        # eliminate effect of zero-padding
        tmp_corr = my_centered(tmp_corr, len(lags))

        # store lags of interest
        tmp_corr = tmp_corr[store_lags]

        pairs_win.append(pair)
        corr.append(tmp_corr)

    corr = np.array(corr)

    return pairs_win, corr


def xcorr_rot(data_fft, stations_win, corr_cmp, P):
    lags = correlation_lags(P.par['corr_npts'], P.par['corr_npts'])
    maxlag = int(P.par["maxlag"] / P.par['dt'])  # maxlag to store (in samples)
    store_lags = np.where(np.abs(lags) <= maxlag)[0]

    assert(corr_cmp in ["RR", "TT"]),("incorrect component")

    pairs_win = []
    corr = []

    for pair in P.pairs_list:
        sta1, sta2 = pair.split("_")

        flag1 = sta1 in stations_win["E"] and sta1 in stations_win["N"]
        flag2 = sta2 in stations_win["E"] and sta2 in stations_win["N"]

        if not flag1 or not flag2:
            continue

        i1_E = stations_win["E"].index(sta1)
        i1_N = stations_win["N"].index(sta1)

        i2_E = stations_win["E"].index(sta2)
        i2_N = stations_win["N"].index(sta2)

        if corr_cmp == "RR":
            w1 = np.cos(np.deg2rad(P.pairs[pair]["az"]))
            w2 = np.sin(np.deg2rad(P.pairs[pair]["az"]))
        elif corr_cmp == "TT":
            w1 = -np.sin(np.deg2rad(P.pairs[pair]["az"]))
            w2 = np.cos(np.deg2rad(P.pairs[pair]["az"]))

        fft_sta1 = w1 * data_fft["N"][i1_N,:] + w2 * data_fft["E"][i1_E,:]
        fft_sta2 = w1 * data_fft["N"][i2_N,:] + w2 * data_fft["E"][i2_E,:]

        # linear cross-correlation of sta1 with sta2 as in equation 11 of
        # Tromp et al. 2010
        tmp_corr = fft_sta1 * np.conj(fft_sta2)

        # convert to time domain, this results in [pos_lags, neg_lags]
        tmp_corr = np.real(np.fft.irfft(tmp_corr, P.par['corr_nfft'],
                                        norm="backward"))

        # switch second and first halves of corr to obtain [neg_lags, pos_lags]
        tmp_corr = np.fft.fftshift(tmp_corr)

        # eliminate effect of zero-padding
        tmp_corr = my_centered(tmp_corr, len(lags))

        # store lags of interest
        tmp_corr = tmp_corr[store_lags]

        pairs_win.append(pair)
        corr.append(tmp_corr)

    corr = np.array(corr)

    return pairs_win, corr
