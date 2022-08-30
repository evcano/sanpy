import itertools
import numpy as np
import os
import shutil
import sys
from mpi4py import MPI
from obspy import read, UTCDateTime, Stream
from obspy.io.sac.sactrace import SACTrace

from sanpy.base.functions import (check_missing_logs,
                                  distribute_objects,
                                  write_log)

from sanpy.base.project_functions import load_project
from sanpy.correlation.functions import *


"""
st, fft index equals the station given by "win_stations[index]"
corr index equals the pair given by "win_pairs[index]"
corr_day index equals the pair given by "all_pairs[index]"

Noise correlations are defined as in Tromp et al. 2010:
    c^ab = s^a(w) * complex_conjugate(s^b(w))

The acausal branch shows waves from a to b
The causal branch shows waves from b to a
"""

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

# distribute jobs
project_path = sys.argv[1]

P = load_project(project_path)

pending_days = check_missing_logs(P.par['log_path'], P.data_span)

if len(pending_days) == 0:
    print('No more days to correlate')
    comm.Abort()

days_to_correlate = distribute_objects(pending_days, nproc, myrank)
ndays_proc = len(days_to_correlate)

if myrank == 0:
    print('Each rank will correlate ~{} days'.format(ndays_proc))

# do jobs
all_stations = P.stations.index.to_list()
nstations = len(all_stations)

all_pairs = P.pairs.index.to_list()
npairs = len(all_pairs)

# loop over days
for day in days_to_correlate:
    corr_day = np.zeros((npairs, P.par['out_npts']))
    corr_counter = np.zeros(npairs)

    psd_day = np.zeros((nstations, P.par['nfft']//2+1))
    psd_counter = np.zeros(nstations)

    st = Stream()

    # read data
    day_waveforms_name = P.waveforms_paths_perday[day]

    if not day_waveforms_name:
        print('no data for current day')
        write_log(P.par['log_path'], day, ['none'])
        continue

    for wf_file in day_waveforms_name:
        st += read(os.path.join(P.par['data_path'], wf_file),
                   format=P.par['data_format'])

    # determine thresholds for transient signals
    if P.par['remove_transient_signals']:
        transient_thresholds = transient_signal_thresholds(st,
                                                           P.par['corr_dur'],
                                                           P.par['corr_overlap'],
                                                           thr=P.par['transient_thr'])

    # compute correlations on a certain window of the day
    for win_st in st.slide(P.par['corr_dur'], P.par["corr_overlap"]):

        # remove windows with time gaps and transient signals
        for tr in win_st:
            if tr.stats.npts != P.par['corr_npts']:
                win_st.remove(tr)
            elif P.par['remove_transient_signals']:
                if remove_transient_signal(tr, transient_thresholds):
                    win_st.remove(tr)

        # list available station pairs
        win_stations = ["{}.{}".format(tr.stats.network, tr.stats.station)
                        for tr in win_st]

        win_pairs = itertools.combinations(win_stations, 2)
        win_pairs = ["%s_%s" % (p[0], p[1]) for p in win_pairs]

        if not win_pairs:
            print('no data for current window')
            continue

        win_st.detrend("demean")
        win_st.taper(0.04)
        data = np.asarray([tr.data for tr in win_st])

        fft = np.fft.rfftn(data,
                           s=[P.par['nfft']],
                           axes=[1],
                           norm="backward")

        if P.par['whitening']:
            single_psd = compute_single_psd(data, P.par)
            fft = uniform_spectral_whitening(fft, single_psd)

        corr = xcorr(fft, win_stations, win_pairs, P.par)
        corr = uniform_time_normalization(corr)

        # stack correlations
        for i, pair in enumerate(win_pairs):
            idx = all_pairs.index(pair)
            corr_day[idx, :] += corr[i, :]
            corr_counter[idx] += 1

        # stack psd
        if P.par['save_psd']:
            fft = np.square(np.real(fft))

            for i, sta in enumerate(win_stations):
                idx = all_stations.index(sta)
                psd_day[idx, :] += fft[i, :]
                psd_counter[idx] += 1

    # compute and save psd of the day
    if P.par['save_psd']:
        psd_counter[psd_counter == 0] = 1
        psd_day = np.divide(psd_day, psd_counter[:, None])
        psd_day = psd_day.astype('float32')

        tmp = np.nansum(np.abs(psd_day), axis=1)
        idx = np.where(tmp != 0)[0]

        for i in idx:
            psd_file = '{}_{}'.format(all_stations[i], day)

            np.save(os.path.join(P.par['psd_path'], all_stations[i],
                                 psd_file), psd_day[i, :])

    # compute correlations of the day
    corr_counter[corr_counter == 0] = 1  # to avoid divison by zero
    corr_day = np.divide(corr_day, corr_counter[:, None])
    corr_day = corr_day.astype("float32")

    # save correlations of the day
    day_obj = UTCDateTime(day)
    tmp = np.nansum(np.abs(corr_day), axis=1)
    idx = np.where(tmp != 0)[0]
    day_pairs = []

    for i in idx:
        pair = all_pairs[i]
        day_pairs.append(pair)
        s1, s2 = pair.split("_")

        header = {
            "kstnm": s2,
            "kcmpnm": P.stations['cmp'][s2],
            "stla": P.stations['lat'][s2],
            "stlo": P.stations['lon'][s2],
            "stel": P.stations['elv'][s2],
            "kevnm": s1,
            "evla": P.stations['lat'][s1],
            "evlo": P.stations['lon'][s1],
            "evdp": P.stations['elv'][s1],
            "lcalda": 1,
            "dist": P.pairs['dis'][pair],
            "nzyear": day_obj.year,
            "nzjday": day_obj.julday,
            "nzhour": day_obj.hour,
            "nzmin": day_obj.minute,
            "nzsec": day_obj.second,
            "nzmsec": day_obj.microsecond,
            "delta": P.par["dt"],
            "b": 0.0}

        tr = SACTrace(data=corr_day[i, :], **header)
        filename = '{}_{}.{}'.format(pair, day, P.par['output_format'])
        tr.write(os.path.join(P.par['output_path'], pair, filename))

    write_log(P.par['log_path'], day, day_pairs)

    ndays_proc -= 1

    if myrank == 0:
        print('~{} days left per core'.format(ndays_proc))

if myrank == 0:
    shutil.copy(project_path, P.par['output_path'])
