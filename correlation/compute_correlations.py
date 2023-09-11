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

pending_days = check_missing_logs(log_path=P.par['log_path'],
                                  log_names=P.data_span)
if len(pending_days) == 0:
    print('No more days to correlate')
    comm.Abort()

days_to_correlate = distribute_objects(pending_days, nproc, myrank)
ndays_proc = len(days_to_correlate)
if myrank == 0:
    print('Each rank will correlate ~{} days'.format(ndays_proc))

# do jobs
stations_list = P.stations_list
pairs_list = P.pairs_list
unique_pairs_list = P.unique_pairs_list

# loop over days
for day in days_to_correlate:
    # read all data
    day_waveforms = P.waveforms_paths_perday[day]

    if not day_waveforms:
        print('no data for current day')
        write_log(P.par['log_path'], day, ['none'])
        continue
    else:
        st = Stream()
        for wf_file in day_waveforms:
            st += read(os.path.join(P.par['data_path'], wf_file),
                       format=P.par['data_format'])

    # determine which data components are available
    data_cmpts_day = []
    for cmp in P.par["data_cmpts"]:
        if st.select(component=cmp):
            data_cmpts_day.append(cmp)

    # determine which correlation components can be computed
    corr_cmpts_day = []
    for cmp in P.par["corr_cmpts"]:
        cmp1, cmp2 = cmp
        if cmp1 in data_cmpts_day and cmp2 in data_cmpts_day:
            corr_cmpts_day.append(cmp)
    if not corr_cmpts_day:
        print("missing data components; cannot compute requested corr")
        continue

    # determine thresholds to discard transient signals
    if P.par['remove_transient_signals']:
        tr_thresholds = transient_signal_thresholds(st,
                                                    P.par['corr_dur'],
                                                    P.par['corr_overlap'],
                                                    thr=P.par['transient_thr'])

    # declare arrays to store the correlations/psd of the day
    corr_day = {}
    count_corr = {}
    for cmp in corr_cmpts_day:
        if cmp in ["NN","EE","ZZ"]:
            n = len(unique_pairs_list)
        else:
            n = len(pairs_list)
        corr_day[cmp] = np.zeros((n, P.par['out_npts']))
        count_corr[cmp] = np.zeros(n)

    if P.par["save_psd"]:
        psd_day = {}
        count_psd = {}
        for cmp in data_cmpts_day:
            psd_day[cmp] = np.zeros((len(stations_list), P.par['nfft']//2+1))
            count_psd[cmp] = np.zeros(len(stations_list))

    # slide a window over the data
    for st_win in st.slide(P.par['corr_dur'], P.par["corr_overlap"]):
        # remove traces with time gaps or transient signals
        for tr in st_win:
            if tr.stats.npts != P.par['corr_npts']:
                st_win.remove(tr)
            elif P.par['remove_transient_signals']:
                remove_flag = remove_transient_signal(tr, tr_thresholds)
                if remove_flag:
                    st_win.remove(tr)

        # check that there is data
        if not st_win:
            print("no data for window")
            continue

        # determine which data components are available
        data_cmpts_win = []
        for cmp in data_cmpts_day:
            if st_win.select(component=cmp):
                data_cmpts_win.append(cmp)

        # determine which correlation components can be computed
        corr_cmpts_win = []
        for cmp in corr_cmpts_day:
            cmp1, cmp2 = cmp
            if cmp1 in data_cmpts_win and cmp2 in data_cmpts_win:
                corr_cmpts_win.append(cmp)
        if not corr_cmpts_win:
            print("missing data components; cannot compute requested corr")
            continue

        # remove the mean and taper the window
        st_win.detrend("demean")
        st_win.taper(0.04)

        # for each data component, compute data fft and estimate noise psd
        data = {}
        data_fft = {}
        noise_psd = {}
        stations_win = {}
        pairs_win = {}

        for cmp in data_cmpts_win:
            st_win_cmp = st_win.select(component=cmp)

            stations_win[cmp] = [f"{tr.stats.network}.{tr.stats.station}"
                                 for tr in st_win_cmp]
            stations_win[cmp].sort()

            data[cmp] = np.asarray([tr.data for tr in st_win_cmp])

            data_fft[cmp] = np.fft.rfftn(data[cmp],
                                         s=[P.par["nfft"]],
                                         axes=[1],
                                         norm="backward")

            if P.par['whitening']:
                noise_psd[cmp] = compute_single_psd(data[cmp], P.par)

        # apply spectral whitening
        if P.par["whitening"]:
            for cmp in data_cmpts_win:
                norm_psd = noise_psd[cmp]
                if cmp in ["N", "E"]:
                    if "NE" in corr_cmpts_win or "EN" in corr_cmpts_win:
                        norm_psd = (noise_psd["N"]+noise_psd["E"]) / 2.0

                data_fft[cmp] = uniform_spectral_whitening(data_fft[cmp],
                                                           norm_psd)

        # compute and stack noise correlations for all components
        for cmp in corr_cmpts_win:
            # create pairs for ZZ,NN,EE,ZN,EN, etc
            cmp1, cmp2 = cmp
            if cmp1 == cmp2:
                # ABC -> AA AB AC BB BC CC
                pairs_win[cmp] = itertools.combinations_with_replacement(
                    stations_win[cmp1],2)
            else:
                # AB; XY -> AX AY BX BY
                pairs_win[cmp] = itertools.product(stations_win[cmp1],
                                                   stations_win[cmp2])

            pairs_win[cmp] = [f"{s[0]}_{s[1]}" for s in pairs_win[cmp]]
            pairs_win[cmp].sort()

            corr = xcorr(data_fft, stations_win, pairs_win, cmp, P.par)
            corr = uniform_time_normalization(corr)

            # stack correlations
            if cmp in ["NN","EE","ZZ"]:
                pairs_mapping = unique_pairs_list
            else:
                pairs_mapping = pairs_list

            for i, pair in enumerate(pairs_win[cmp]):
                idx = pairs_mapping.index(pair)
                corr_day[cmp][idx, :] += corr[i, :]
                count_corr[cmp][idx] += 1

        # stack psd
        if P.par['save_psd']:
            for cmp in data_cmpts_win:
                data_fft[cmp] = np.square(np.real(data_fft[cmp]))
                for i, sta in enumerate(stations_win[cmp]):
                    idx = stations_list.index(sta)
                    psd_day[cmp][idx, :] += data_fft[cmp][i, :]
                    count_psd[cmp][idx] += 1

    for cmp in corr_cmpts_day:
        # compute correlations of the day
        count_corr[cmp][count_corr[cmp]== 0] = 1  # to avoid divison by zero
        corr_day[cmp] = np.divide(corr_day[cmp], count_corr[cmp][:, None])
        corr_day[cmp] = corr_day[cmp].astype("float32")

        # save correlations of the day
        day_obj = UTCDateTime(day)
        tmp = np.nansum(np.abs(corr_day[cmp]), axis=1)
        idx = np.where(tmp != 0)[0]
        day_pairs = []

        if cmp in ["NN","EE","ZZ"]:
            pairs_mapping = unique_pairs_list
        else:
            pairs_mapping = pairs_list

        for i in idx:
            pair = pairs_mapping[i]
            day_pairs.append(pair)
            s1, s2 = pair.split("_")

            header = {
                "kstnm": s2,
                "kcmpnm": cmp,
                "stla": P.stations[s2]['lat'],
                "stlo": P.stations[s2]['lon'],
                "stel": P.stations[s2]['elv'],
                "kevnm": s1,
                "evla": P.stations[s1]['lat'],
                "evlo": P.stations[s1]['lon'],
                "evdp": P.stations[s1]['elv'],
                "lcalda": 1,
                "dist": P.pairs[pair]['dis'],
                "nzyear": day_obj.year,
                "nzjday": day_obj.julday,
                "nzhour": day_obj.hour,
                "nzmin": day_obj.minute,
                "nzsec": day_obj.second,
                "nzmsec": day_obj.microsecond,
                "delta": P.par["dt"],
                "b": 0.0}

            tr = SACTrace(data=corr_day[cmp][i, :], **header)
            filename = f"{pair}_{cmp}_{day}.{P.par['output_format']}"
            tr.write(os.path.join(P.par['output_path'], pair, filename))

    # compute and save psd of the day
    if P.par['save_psd']:
        for cmp in data_cmpts_day:
            count_psd[cmp][count_psd[cmp] == 0] = 1
            psd_day[cmp] = np.divide(psd_day[cmp], count_psd[cmp][:, None])
            psd_day[cmp] = psd_day[cmp].astype('float32')

            tmp = np.nansum(np.abs(psd_day[cmp]), axis=1)
            idx = np.where(tmp != 0)[0]

            for i in idx:
                psd_file = f"{stations_list[i]}_{cmp}_{day}"
                psd_file = os.path.join(P.par["psd_path"], stations_list[i],
                                        psd_file)
                np.save(psd_file, psd_day[cmp][i, :])

    write_log(P.par['log_path'], day, day_pairs)
    ndays_proc -= 1
    if myrank == 0:
        print('~{} days left per core'.format(ndays_proc))

if myrank == 0:
    shutil.copy(project_path, P.par['output_path'])
