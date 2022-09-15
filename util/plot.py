import matplotlib.pyplot as plt
import numpy as np
import os
from obspy import read, Stream
from sanpy.util.apparent_velocity import compute_apparent_velocity


def plot_correlations(data_path, data_format, pairs=None, maxtime=None,
                      bandpass=None, global_normalization=False, yaxis=None,
                      amplitude_only=False, apparent_velocity=False):

    # read data
    st = Stream()

    if len(pairs) > 0:
        files = ['{}.{}'.format(x, data_format) for x in pairs]

        for f in files:
            stpath = os.path.join(data_path, f)

            if os.path.isfile(stpath):
                st += read(stpath, format=data_format)
    else:
        stpath = os.path.join(data_path, '*')
        st += read(stpath, format=data_format)

    ntr = len(st)

    # cut data
    if maxtime:
        maxlag = ((st[0].stats.npts - 1) / 2) * st[0].stats.delta

        for i in range(0, ntr):
            st[i] = st[i].slice(st[i].stats.starttime + maxlag - maxtime,
                                st[i].stats.starttime + maxlag + maxtime)

    maxtime = ((st[0].stats.npts - 1) / 2) * st[0].stats.delta
    lags = np.linspace(-maxtime, maxtime, st[0].stats.npts)

    # filter data and normalize
    if bandpass:
        st.taper(0.04)
        st.filter('bandpass',
                  freqmin=bandpass[0],
                  freqmax=bandpass[1],
                  corners=2,
                  zerophase=True)

    if global_normalization:
        st.normalize(global_max=True)
    else:
        st.normalize(global_max=False)

    # sort data according to interstation distance
    distances = []
    for tr in st:
        distances.append(tr.stats.sac.dist)

    idx = np.argsort(np.array(distances))
    distances = np.sort(distances)

    data = np.zeros((ntr, st[0].stats.npts))
    for i, j in enumerate(idx):
        data[i, :] = st[j].data

    # estimate apparent velocity
    if apparent_velocity:
        pos_win = np.zeros(lags.shape)
        pos_win[np.where(lags >= 0.0)] = 1.0

        pos_c1, pos_c2 = compute_apparent_velocity(data*pos_win, lags,
                                                   distances)

        neg_c1, neg_c2 = compute_apparent_velocity(data*pos_win[::-1], lags,
                                                   distances)

    # setup figure
    fig, ax = plt.subplots()

    ax.set_title('Noise correlations')
    ax.set_xlabel('Lag [s]')

    if yaxis and yaxis == 'dis' and amplitude_only is False:
        ax.set_ylabel('Interstation distance [km]')
    else:
        ax.set_ylabel('Unitless')

    # plot data
    if amplitude_only:
        ax.imshow(data, extent=[lags[0], lags[-1], 0, ntr-1])
    else:
        offset = 0

        for i in range(0, ntr):
            if yaxis and yaxis == 'dis':
                data[i, :] += distances[i]
            else:
                data[i, :] += offset
                offset = np.max(data[i, :])

            ax.plot(lags, data[i, :], c='k', lw=0.5, alpha=0.5)

    print('{} correlations plotted'.format(ntr))

    if apparent_velocity:
        print('Acausal-branch apparent velocity: {} km/s'.format(1.0/neg_c1))
        print('Causal-branch apparent velocity: {} km/s'.format(1.0/pos_c1))

        if yaxis and yaxis == 'dis':
            plt.plot(pos_c1*np.array(distances)+pos_c2, distances, 'r')
            plt.plot(neg_c1*np.array(distances)+neg_c2, distances, 'r')

    plt.show()

    return


def plot_greens(data_path, data_format, pairs=None, maxtime=None,
                bandpass=None, global_normalization=False, yaxis=None,
                amplitude_only=False, apparent_velocity=False):

    # read data
    st = Stream()

    if pairs:
        files = ['{}.{}'.format(x, data_format) for x in pairs]

        for f in files:
            stpath = os.path.join(data_path, f)

            if os.path.isfile(stpath):
                st += read(stpath, format=data_format)
    else:
        stpath = os.path.join(data_path, '*')
        st += read(stpath, format=data_format)

    ntr = len(st)

    # cut maximum time
    if maxtime:
        for i in range(0, ntr):
            st[i] = st[i].slice(st[i].stats.starttime,
                                st[i].stats.starttime + maxtime)

    times = st[0].times()

    # filter data and normalize
    if bandpass:
        st.taper(0.04)
        st.filter('bandpass',
                  freqmin=bandpass[0],
                  freqmax=bandpass[1],
                  corners=2,
                  zerophase=True)

    if global_normalization:
        st.normalize(global_max=True)
    else:
        st.normalize(global_max=False)

    # sort data according to interstation distance
    distances = []
    for tr in st:
        distances.append(tr.stats.sac.dist)

    idx = np.argsort(np.array(distances))
    distances = np.sort(distances)

    data = np.zeros((ntr, st[0].stats.npts))
    for i, j in enumerate(idx):
        data[i, :] = st[j].data

    # estimate apparent velocity
    if apparent_velocity:
        c1, c2 = compute_apparent_velocity(data, times, distances)

    # setup figure
    fig, ax = plt.subplots()

    ax.set_title("Empirical Green's functions")
    ax.set_xlabel('Time [s]')

    if yaxis and yaxis == 'dis' and amplitude_only is False:
        ax.set_ylabel('Interstation distance [km]')
    else:
        ax.set_ylabel('Unitless')

    # plot data
    if amplitude_only:
        ax.imshow(data, extent=[times[0], times[-1], 0, ntr-1])
    else:
        offset = 0

        for i in range(0, ntr):
            if yaxis and yaxis == 'dis':
                data[i, :] += distances[i]
            else:
                data[i, :] += offset
                offset = np.max(data[i, :])

            ax.plot(times, data[i, :], c='k', lw=0.5, alpha=0.5)

    print("{} empirical Green's functions plotted".format(len(idx)))

    if apparent_velocity:
        print('Apparent velocity: {} km/s'.format(1.0/c1))

        if yaxis and yaxis == 'dis':
            plt.plot(c1*np.array(distances)+c2, distances, 'r')

    plt.show()

    return
