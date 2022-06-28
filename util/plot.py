import matplotlib.pyplot as plt
import numpy as np
import os
from obspy import read, Stream


def plot_correlations(data_path, data_format, pairs=None, maxtime=None,
                      bandpass=None, global_normalization=False, yaxis=None):

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

    # cut data
    if maxtime:
        maxlag = ((st[0].stats.npts - 1) / 2) * st[0].stats.delta

        for i in range(0, len(st)):
            st[i] = st[i].slice(st[i].stats.starttime + maxlag - maxtime,
                                st[i].stats.starttime + maxlag + maxtime)

    maxtime = (st[0].stats.npts - 1) / 2
    maxtime *= st[0].stats.delta
    lags = np.linspace(-maxtime, maxtime, st[0].stats.npts)

    # filter data and normalize to global maximum
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

    # get data order according to interstation distance
    distances = []

    for tr in st:
        distances.append(tr.stats.sac.dist)

    idx = np.argsort(np.array(distances))

    # setup figure
    fig, ax = plt.subplots()

    ax.set_title('Noise correlations')
    ax.set_xlabel('Lag [s]')

    if yaxis and yaxis == 'dis':
        ax.set_ylabel('Interstation distance [km]')
    else:
        ax.set_ylabel('Unitless')

    # plot data
    offset = 0

    for i in idx:
        if yaxis and yaxis == 'dis':
            data = st[i].data + st[i].stats.sac.dist
        else:
            data = st[i].data + offset
            offset = np.max(data)

        ax.plot(lags, data, c='k', lw=0.5, alpha=0.5)

    print('{} correlations plotted'.format(len(idx)))

    plt.show()

    return


def plot_greens(data_path, data_format, pairs=None, maxtime=None,
                bandpass=None, global_normalization=False, yaxis=None):

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

    # cut maximum time
    if maxtime:
        for i in range(0, len(st)):
            st[i] = st[i].slice(st[i].stats.starttime,
                                st[i].stats.starttime + maxtime)

    times = st[0].times()

    # filter data and normalize to global maximum
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

    # get data order according to interstation distance
    distances = []

    for tr in st:
        distances.append(tr.stats.sac.dist)

    idx = np.argsort(np.array(distances))

    # setup figure
    fig, ax = plt.subplots()

    ax.set_title("Empirical Green's functions")
    ax.set_xlabel('Time [s]')

    if yaxis and yaxis == 'dis':
        ax.set_ylabel('Interstation distance [km]')
    else:
        ax.set_ylabel('Unitless')

    # plot data
    offset = 0

    for i in idx:
        if yaxis and yaxis == 'dis':
            data = st[i].data + st[i].stats.sac.dist
        else:
            data = st[i].data + offset
            offset = np.max(data)

        ax.plot(times, data, c='k', lw=0.5, alpha=0.5)

    print("{} empirical Green's functions plotted".format(len(idx)))

    plt.show()

    return
