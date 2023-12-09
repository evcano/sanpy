import numpy as np
import os
import scipy.fft as sf


def check_sample_aligment(st):
    """
    if tr.starttime is not a multiple of i*dt, i=0,1,..npts, apply
    shift in frequency domain
    """

    for tr in st:
        start_time = tr.stats.starttime.datetime.microsecond * 1.0e-6  # sec
        dt = tr.stats.delta

        # if start_time is multiple of dt (the residual of start_time/dt is
        # zero) then shift = 0
        shift = np.mod(start_time, dt)

        if shift == 0 or (dt - shift) <= np.finfo(float).eps:
            continue
        else:
            if shift <= (dt / 2.):
                shift = - shift  # left shift
            else:
                shift = dt - shift  # right shift

            tr.detrend(type="demean")
            tr.detrend(type="linear")
            tr.taper(max_percentage=None, max_length=1.0)

            nfft = sf.next_fast_len(tr.stats.npts)
            freq_axis = sf.fftfreq(nfft, d=dt)
            fft = sf.fft(tr.data, n=nfft)
            fft = fft * np.exp(1j * 2. * np.pi * freq_axis * shift)
            fft = fft.astype(np.complex64)
            tmp = sf.ifft(fft, n=nfft)
            tr.data = np.real(tmp[:len(tr.data)])
            tr.stats.starttime += shift
            del fft, freq_axis, tmp

    return st


def fill_gaps(st, par):
    # NOTE: Gaps at the start (00:00) and end (23:59) of the record day cannot
    # be filled as we need the records of the previous and following day.

    st = st.split()  # we split traces that contain gaps
    st.sort(keys=["starttime"])  # sort them by increasing time

    # check that all traces correspond to the same station and channel
    # set the same data type for all traces
    for tr in st:
        if tr.stats.network != st[0].stats.network:
            raise Exception
        elif tr.stats.station != st[0].stats.station:
            raise Exception
        elif tr.stats.channel != st[0].stats.channel:
            raise Exception
        tr.data = tr.data.astype("float64")

    ntr = len(st)

    if ntr > 1:
        i = 0

        # if there is a trace at a later time of the current trace,
        # keep filling gaps, i.e., stop once we are at the last
        # trace of the stream
        while i < (ntr - 1):
            t1 = st[i].stats['endtime']
            t2 = st[i+1].stats['starttime']
            dif = t2 - t1  # seconds

            if dif <= par['gap_tolerance']:
                st[i] = st[i].__add__(st[i+1],
                                      method=1,
                                      fill_value="interpolate")
                st.remove(st[i+1])
            else:
                i += 1

            ntr = len(st)

    return st


def get_pending_waveforms(output_path, waveforms_paths):
    """
    output_path: path where processed waveforms are stored
    waveforms_paths: list with elements as: net/sta/waveform_file

    """
    pending_waveforms = []

    for path in waveforms_paths:
        tmp = os.path.join(output_path, path)

        if not os.path.isfile(tmp):
            pending_waveforms.append(path)

    return pending_waveforms


def preprocess(st, par, inv=[]):
    for tr in st:
        # delete short traces
        if (tr.stats.npts*tr.stats.delta) < (2.0 * par["taper_length"]):
            st.remove(tr)
            continue

        tr.detrend(type="demean")
        tr.detrend(type="linear")
        tr.taper(max_percentage=None,
                 max_length=par["taper_length"])

        tr.filter('lowpass', freq=par['lowpassfq'], zerophase=True, corners=8)

        tr.data = np.array(tr.data)
        tr.interpolate(method="lanczos",
                       sampling_rate=par["new_sampling_rate"],
                       a=1.0)

        tr.stats.sampling_rate = par["new_sampling_rate"]

        if par["remove_response"]:
            if not inv:
                print("Inventory was not provided.")
                raise Exception

            # if there is more than 1 response for the channel,
            # remove_response searches for the correct one
            prefilter = par['response_prefilter'][tr.stats.network]
            waterlevel = par['response_waterlevel'][tr.stats.network]

            tr.remove_response(inventory=inv,
                               pre_filt=prefilter,
                               water_level=waterlevel,
                               output=par["units"],
                               plot=False)

    return st
