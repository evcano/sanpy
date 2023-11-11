import itertools
import numpy as np
import os
import shutil
import sys
from mpi4py import MPI
from obspy import read, UTCDateTime, Stream
from obspy.io.sac.sactrace import SACTrace
from scipy.stats import scoreatpercentile

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
nstations = len(stations_list)

pairs_list = P.pairs_list
npairs = len(pairs_list)

# read bad windows
if P.par['bad_windows']:
    bad_win = {}
    for sta in stations_list:
        file_ = os.path.join(P.par['bad_windows'], f"{sta.upper()}_OUTLIERS.dat")
        bad_win[sta] = read_bad_windows(file_)

# loop over days
for day in days_to_correlate:
    # read all data
    waveforms_files = P.waveforms_paths_perday[day]

    if not waveforms_files:
        print(f"{day} contains no data")
        write_log(log_path=P.par['log_path'], log_name=day, log=['none'])
        continue
    else:
        st = Stream()
        for file_ in waveforms_files:
            st += read(os.path.join(P.par['data_path'], file_),
                       format=P.par['data_format'])

    # declare arrays to store the correlations/psd of the day
    corr_day = {}
    count_corr = {}
    for cmp in P.par["corr_cmpts"]:
        corr_day[cmp] = np.zeros((npairs, P.par['save_npts']))
        count_corr[cmp] = np.zeros(npairs)

    if P.par["save_psd"]:
        psd_day = {}
        count_psd = {}
        for cmp in P.par["data_cmpts"]:
            psd_day[cmp] = np.zeros((nstations, P.par['corr_nfft']//2+1))
            count_psd[cmp] = np.zeros(nstations)

    # slide a window over the data
    for st_win in st.slide(P.par['corr_dur'],
                           P.par['corr_dur']-P.par["corr_overlap"]):

        # remove traces with time gaps or bad windows
        for tr in st_win:
            if tr.stats.npts != P.par['corr_npts']:
                st_win.remove(tr)
            elif P.par['bad_windows']:
                tr_code = f"{tr.stats.network}.{tr.stats.station}"
                tr_start = tr.stats.starttime
                tr_end = tr.stats.endtime
                if day in bad_win[tr_code].keys():
                    for wstart in bad_win[tr_code][day]:
                        wend = wstart + 3600.0
                        flag1 = tr_start <= wstart < tr_end
                        flag2 = tr_start < wend <= tr_end
                        if flag1 or flag2:
                            st_win.remove(tr)
                            break

        # check that there is data
        if not st_win:
            print("no data for window")
            continue

        # for each component, compute fft and apply spectral whitening
        data = {}
        data_fft = {}
        stations_win = {}

        for cmp in P.par["data_cmpts"]:
            st_win_cmp = st_win.select(component=cmp)

            if not st_win_cmp:
                continue

            st_win_cmp.detrend("demean")
            st_win_cmp.taper(0.05)

            data[cmp] = np.asarray([tr.data for tr in st_win_cmp])

            data_fft[cmp] = np.fft.rfftn(data[cmp],
                                         s=[P.par["corr_nfft"]],
                                         axes=[1],
                                         norm="backward")

            stations_win[cmp] = [f"{tr.stats.network}.{tr.stats.station}"
                                 for tr in st_win_cmp]

            if P.par["whitening"]:
                norm_spec = compute_single_psd(data[cmp], P.par)
                data_fft[cmp] = np.divide(data_fft[cmp], norm_spec)

                for q in range(0, data_fft[cmp].shape[0]):
                    tmp = data_fft[cmp][q,:]
                    imin = scoreatpercentile(tmp,5)
                    imax = scoreatpercentile(tmp,95)
                    notout = np.where((tmp>=imin)&(tmp<=imax))
                    cval = np.max(np.abs(tmp[notout]))
                    data_fft[cmp][q,:] = np.clip(data_fft[cmp][q,:],-cval,cval)

        # determine available correlation components
        avail_data_cmpts = list(data.keys())
        avail_corr_cmpts = []
        for cmp in P.par["corr_cmpts"]:
            if cmp in ["EE", "NN", "ZZ"]:
                if cmp[0] in avail_data_cmpts:
                    avail_corr_cmpts.append(cmp)
            elif cmp in ["RR", "TT"]:
                if "E" in avail_data_cmpts and "N" in avail_data_cmpts:
                    avail_corr_cmpts.append(cmp)

        # compute and stack noise correlations
        corr = {}
        pairs_win = {}

        for cmp in avail_corr_cmpts:
            if cmp in ["EE", "NN", "ZZ"]:
                pairs_win[cmp], corr[cmp] = xcorr(data_fft,
                                                  stations_win,
                                                  cmp,
                                                  P)
            elif cmp in ["RR", "TT"]:
                pairs_win[cmp], corr[cmp] = xcorr_rot(data_fft,
                                                      stations_win,
                                                      cmp,
                                                      P)

        # time normalization
        max_amps = []
        for cmp in avail_corr_cmpts:
            if corr[cmp].any():
                x = np.max(np.abs(corr[cmp]), axis=1)
                max_amps.extend(x)
        norm_fact = np.percentile(max_amps, 95)
        for cmp in avail_corr_cmpts:
            if corr[cmp].any():
                corr[cmp] = np.divide(corr[cmp],norm_fact)

        # stack correlations
        for cmp in avail_corr_cmpts:
            for i, pair in enumerate(pairs_win[cmp]):
                j = pairs_list.index(pair)
                corr_day[cmp][j,:] += corr[cmp][i,:]
                count_corr[cmp][j] += 1

        # stack psd
        if P.par['save_psd']:
            for cmp in avail_data_cmpts:
                data_fft[cmp] = np.square(np.real(data_fft[cmp]))
                for i, sta in enumerate(stations_win[cmp]):
                    j = stations_list.index(sta)
                    psd_day[cmp][j,:] += data_fft[cmp][i,:]
                    count_psd[cmp][j] += 1

    for cmp in P.par["corr_cmpts"]:
        # normalize correlations of the day to compute the average
        count_corr[cmp][count_corr[cmp]== 0] = 1  # to avoid divison by zero
        corr_day[cmp] = np.divide(corr_day[cmp], count_corr[cmp][:, None])

        # save correlations of the day
        corr_day[cmp] = corr_day[cmp].astype("float32")
        tmp = np.nansum(np.abs(corr_day[cmp]), axis=1)
        idx = np.where(tmp != 0)[0]

        day_obj = UTCDateTime(day)
        day_pairs = []

        for i in idx:
            pair = pairs_list[i]
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
            tr.write(os.path.join(P.par['corr_path'], cmp, pair, filename))

    # compute and save psd of the day
    if P.par['save_psd']:
        for cmp in P.par["data_cmpts"]:
            count_psd[cmp][count_psd[cmp] == 0] = 1
            psd_day[cmp] = np.divide(psd_day[cmp], count_psd[cmp][:, None])

            psd_day[cmp] = psd_day[cmp].astype('float32')
            tmp = np.nansum(np.abs(psd_day[cmp]), axis=1)
            idx = np.where(tmp != 0)[0]

            for i in idx:
                psd_file = f"{stations_list[i]}_{cmp}_{day}"
                psd_file = os.path.join(P.par["psd_path"], cmp, stations_list[i],
                                        psd_file)
                np.save(psd_file, psd_day[cmp][i, :])

    write_log(P.par['log_path'], day, day_pairs)
    ndays_proc -= 1
    if myrank == 0:
        print('~{} days left per core'.format(ndays_proc))

if myrank == 0:
    shutil.copy(project_path, P.par['corr_path'])
