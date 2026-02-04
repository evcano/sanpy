import matplotlib.pyplot as plt
import numpy as np
import os
from obspy import read, Stream, UTCDateTime
from glob import glob


def _cut_correlations(st, maxlag):
    ntr = len(st)
    branchtime = ((st[0].stats.npts - 1) / 2) * st[0].stats.delta

    if maxlag:
        for i in range(0, ntr):
            st[i] = st[i].slice(st[i].stats.starttime + branchtime - maxlag,
                                st[i].stats.starttime + branchtime + maxlag)

    branchtime = ((st[0].stats.npts - 1) / 2) * st[0].stats.delta
    lags = np.linspace(-branchtime, branchtime, st[0].stats.npts)

    return st, lags


def _cut_waveforms(st, maxtime):
    if maxtime:
        for i in range(0, ntr):
            st[i] = st[i].slice(st[i].stats.starttime,
                                st[i].stats.starttime + maxtime)

    times = st[0].times()

    return st, times


def _filter_data(st, bandpass):
    st.detrend("linear")
    st.detrend("demean")
    st.taper(0.1)

    if bandpass:
        st.filter('bandpass',
                  freqmin=bandpass[0],
                  freqmax=bandpass[1],
                  corners=4,
                  zerophase=True)

    return st


def _read_data(data_path, data_format, cmp, pairs):
    st = Stream()

    if pairs and len(pairs) > 0:
        files = [f"{x}_{cmp}.{data_format}" for x in pairs]
        files = [os.path.join(data_path, f) for f in files]
        for f in files:
            if os.path.isfile(f):
                st += read(f, format=data_format)
    else:
        stpath = os.path.join(data_path, '*')
        st += read(stpath, format=data_format)

    ntr = len(st)

    if ntr == 0:
        print("no data")
        return None, None

    return st, ntr


def _st_to_array(st):
    """ data is sorted by interstation distance """
    ntr = len(st)
    data = np.zeros((ntr, st[0].stats.npts))
    distances = []

    for tr in st:
        distances.append(tr.stats.sac.dist)

    idx = np.argsort(np.array(distances))

    for i, j in enumerate(idx):
        data[i, :] = st[j].data

    return data, distances


def plot_correlations(data_path, data_format, cmp, pairs=None,
    bandpass=None, maxlag=None, global_normalization=False, avel=None,
    yaxis='idx', gain=1.0, lw=1.0, alpha=1.0,
    wiggles=False, amplitude_only=False, showfig=True, ax=None):

    # prepare data
    st, ntr = _read_data(data_path, data_format, cmp, pairs)
    st = _filter_data(st, bandpass)
    st, lags = _cut_correlations(st, maxlag)
    st.normalize(global_max=global_normalization)

    # plot data
    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if amplitude_only:
        data, _ = _st_to_array(st)
        ax.imshow(data, extent=[lags[0], lags[-1], 0, ntr-1], origin="lower", aspect="auto", cmap="seismic")
    else:
        for i in range(0, ntr):
            if yaxis == 'dis':
                offset = st[i].stats.sac.dist
            else:
                offset = i

            y = (st[i].data * gain) + offset
            ax.plot(lags, y, c='k', lw=lw, alpha=alpha)

            ## mirror branches
            #ax.plot(lags[np.where(lags>=0)], y[np.where(lags>=0)], c='k', lw=lw, alpha=alpha)
            #ax.plot(-lags[np.where(lags<=0)], y[np.where(lags<=0)], c='b', lw=lw, alpha=alpha)

            if wiggles:
                ax.fill_between(lags, y, y.mean(), where=y>y.mean(),
                    color="k", alpha=alpha, interpolate=True)

    # apparent velocity lines
    if yaxis == 'dis' and avel and amplitude_only == False:
        distances = [tr.stats.sac.dist for tr in st]
        distances = np.sort(np.array(distances))

        for v  in avel:
            tmp = 1/v * np.array(distances)
            ax.plot(tmp, distances, 'g', lw=lw, alpha=0.9)
            ax.plot(-tmp, distances, 'g', lw=lw, alpha=0.9)

            ax.text(0.0, distances[-5], f"{v:.2f} km/s", alpha=0.9, fontsize=9)
            #ax.text(-tmp[-1], distances[-1], f"{v:.2f} km/s", alpha=alpha)

    # figure settings
    ax.set_xlim(-maxlag, maxlag)
    if bandpass:
        ax.set_title(f'{cmp.upper()} noise correlations {bandpass[0]:.2f}-{bandpass[1]:.2f} Hz')
    else:
        ax.set_title(f'{cmp.upper()} noise correlations')
    ax.set_xlabel('Lag [s]')

    if yaxis == 'dis' and amplitude_only is False:
        ax.set_ylabel('Interstation distance [km]')
    else:
        ax.set_ylabel('Unitless')

    if showfig:
        print(f'{ntr} correlations plotted')
        plt.show()
        plt.close()
        return None, None
    else:
        return fig, ax


def plot_daily_correlations(data_path, data_format,
    bandpass=None, maxlag=None, global_normalization=False,
    gain=1.0, lw=1.0, alpha=1.0, wiggles=False, amplitude_only=False,
    showfig=True):

    # read data, sort by date, and get dates
    files = glob(os.path.join(data_path,"*"))
    files.sort()

    st = Stream()
    day_list = []
    dayno_list = []

    for f in files:
        st += read(f, format=data_format)

        basefile = os.path.basename(f)
        sta1, sta2, _, day = basefile.split("_")

        day, _ = day.split(".")
        day = UTCDateTime(day)
        day_list.append(day)

        dayno = day.matplotlib_date - UTCDateTime("1970-01-01 00:00:00").matplotlib_date
        dayno_list.append(dayno)

    ntr = len(st)

    if ntr == 0:
        print("no data")
        return None, None

    # prepare data
    st = _filter_data(st, bandpass)
    st, lags = _cut_correlations(st, maxlag)
    st.normalize(global_max=global_normalization)

    # plot data
    fig, ax = plt.subplots()

    for i in range(0, ntr):
        y = (st[i].data * gain) + dayno_list[i]
        ax.plot(lags, y, c="k", lw=lw, alpha=alpha)

    # figure settings
    dist = st[0].stats.sac.dist
    cmp = st[0].stats.sac.kcmpnm
    title = (
             f"{sta1} - {sta2} {dist:.2f} km \n"
             f"Daily {cmp} noise correlations \n"
             f"{bandpass[0]:.2f} - {bandpass[1]:.2f} Hz"
            )

    ax.set_title(title)
    ax.set_xlabel('Lag [s]')

    yticks = [x for x in dayno_list[::2]]
    ylabels = [f"{x.year}-{x.month:02}-{x.day:02}" for x in day_list[::2]]
    ax.set_yticks(ticks=yticks, labels=ylabels)

    print(f'{ntr} correlations plotted')
    
    if showfig:
        plt.show()
        plt.close()
        return None, None
    else:
        return fig, ax


def plot_greens(data_path, data_format, cmp, pairs=None,
    bandpass=None, maxtime=None, global_normalization=False,
    yaxis='idx', gain=1.0, lw=1.0, alpha=1.0,
    wiggles=False, amplitude_only=False):

    # prepare data
    st, ntr = _read_data(data_path, data_format, cmp, pairs)
    st = _filter_data(st, bandpass)
    st, times = _cut_waveforms(st, maxtime)
    st.normalize(global_max=global_normalization)

    # plot data
    fig, ax = plt.subplots()

    if amplitude_only:
        data, _ = _st_to_array(st)
        ax.imshow(data, extent=[times[0], times[-1], 0, ntr-1])
    else:
        for i in range(0, ntr):
            if yaxis == 'dis':
                offset = st[i].stats.sac.dist
            else:
                offset = i
            y = (st[i].data * gain) + offset
            ax.plot(times, y, c='k', lw=lw, alpha=alpha)

            if wiggles:
                ax.fill_between(times, y, y.mean(), where=y>y.mean(),
                    color="k", alpha=alpha, interpolate=True)

    # figure settings
    ax.set_title(f"Empirical Green's functions {cmp.upper()}")
    ax.set_xlabel('Time [s]')

    if yaxis == 'dis' and amplitude_only is False:
        ax.set_ylabel('Interstation distance [km]')
    else:
        ax.set_ylabel('Unitless')

    print("{} empirical Green's functions plotted".format(len(idx)))
    plt.show()
    plt.close()

    return
