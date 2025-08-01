import matplotlib.pyplot as plt
import itertools
import numpy as np
import os
import shutil
import sys
from mpi4py import MPI
from obspy import read, UTCDateTime, Stream
from obspy.io.sac.sactrace import SACTrace
from obspy.signal.invsim import cosine_sac_taper
from scipy.stats import scoreatpercentile
from scipy.signal import hilbert, windows, convolve

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

# define frequency domain taper
fqax = np.fft.rfftfreq(P.par['corr_nfft'], P.par['dt'])

if P.par['fqtaper']:
    fqcorners = P.par['fqtaper']       
    fqtaper = cosine_sac_taper(freqs=fqax, flimit=fqcorners)
    flatfq_idx = np.where((fqax > fqcorners[1]) & (fqax < fqcorners[2]))
else:
    fqtaper = np.ones(fqax.size)       
    flatfq_idx = np.arange(0, fqax.size)

# read transient signals
if P.par['remove_tsignals']:
    tsignals = read_tsignals(P.par['tsig_path'], stations_list, P.par['data_cmpts'])

# loop over days
for day in days_to_correlate:
    day_obj = UTCDateTime(day)

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

    # declare array to store the correlations of the day
    corr_day = {}
    count_corr = {}
    for cmp in P.par["corr_cmpts"]:
        corr_day[cmp] = np.zeros((npairs, P.par['save_npts']))
        count_corr[cmp] = np.zeros(npairs)

    # slide a window over the data
    for st_win in st.slide(P.par['corr_dur'],
                           P.par['corr_dur']-P.par["corr_overlap"]):

        # check that there is data
        if not st_win:
            print("no data for window")
            continue

        # for each data component compute fft
        data = {}
        data_fft = {}
        stations_win = {}

        for cmp in P.par["data_cmpts"]:
            st_win_cmp = st_win.select(component=cmp)

            # remove traces with time gaps or transient signals
            # transient signals are independently removed per component
            for tr in st_win_cmp:
                if tr.stats.npts != P.par['corr_npts']:
                    st_win_cmp.remove(tr)
                elif P.par['remove_tsignals']:
                    tr_sta = f"{tr.stats.network}.{tr.stats.station}"
                    for tsig in tsignals[tr_sta][cmp]:
                        stime1 = tr.stats.starttime
                        etime1 = tr.stats.endtime
                        stime2 = tsig[0]
                        etime2 = tsig[1]
                        if (stime1 < etime2) and (stime2 < etime1):
                            st_win_cmp.remove(tr)
                            break

            if not st_win_cmp:
                continue

            st_win_cmp.detrend("demean")
            st_win_cmp.taper(0.05)

            stations_win[cmp] = [f"{tr.stats.network}.{tr.stats.station}"
                                 for tr in st_win_cmp]

            data[cmp] = np.asarray([tr.data for tr in st_win_cmp])

            data_fft[cmp] = np.fft.rfftn(data[cmp],
                                         s=[P.par["corr_nfft"]],
                                         axes=[1],
                                         norm="backward")

        # available data components
        avail_data_cmpts = list(data.keys())

        # independently apply spectral whitening to each component
        # and apply frequency taper
        for cmp in avail_data_cmpts:
            if P.par['whitening']:
                for q in range(0, data_fft[cmp].shape[0]):
                    # normalization spectrum
                    norm_spec = np.abs(np.real(data_fft[cmp][q,:]))
                    win_smooth = windows.hann(10)
                    norm_spec = convolve(norm_spec, win_smooth, mode="same")
                    norm_spec /= np.sum(win_smooth)
                    # apply whitening
                    data_fft[cmp][q,:] = np.divide(data_fft[cmp][q,:], norm_spec)
                    # apply frequency taper
                    data_fft[cmp][q,:] *= fqtaper
                    # determine value to clip the spectrum
                    tmp = data_fft[cmp][q,flatfq_idx]
                    imin = scoreatpercentile(tmp, 5)
                    imax = scoreatpercentile(tmp, 95)
                    not_outlier = np.where((tmp >= imin) & (tmp <= imax))
                    rms = tmp[not_outlier].std()

                    # clip spectrum to remove outliers/peaks
                    data_fft[cmp][q,:] = np.clip(data_fft[cmp][q,:], -rms, rms)
            else:
                for q in range(0, data_fft[cmp].shape[0]):
                    data_fft[cmp][q,:] *= fqtaper

        # determine available correlation components
        avail_corr_cmpts = []

        for cmp in P.par["corr_cmpts"]:
            if cmp == "EE" and "E" in avail_data_cmpts:
                avail_corr_cmpts.append(cmp)
            elif cmp == "NN" and "N" in avail_data_cmpts:
                avail_corr_cmpts.append(cmp)
            elif cmp == "ZZ" and "Z" in avail_data_cmpts:
                avail_corr_cmpts.append(cmp)
            elif cmp == "RR" and "E" in avail_data_cmpts and "N" in avail_data_cmpts:
                avail_corr_cmpts.append(cmp)
            elif cmp == "TT" and "E" in avail_data_cmpts and "N" in avail_data_cmpts:
                avail_corr_cmpts.append(cmp)

        if not avail_corr_cmpts:
            continue

        # compute noise correlations
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

        # stack correlations
        for cmp in avail_corr_cmpts:
            for i, pair in enumerate(pairs_win[cmp]):
                j = pairs_list.index(pair)
                # we normalize each correlation by its abs max
                corr[cmp][i,:] /= np.max(np.abs(corr[cmp][i,:]))
                corr_day[cmp][j,:] += corr[cmp][i,:]
                count_corr[cmp][j] += 1

    # save correlations of the day
    for cmp in P.par["corr_cmpts"]:
        day_pairs = []

        for i, pair in enumerate(pairs_list):
            if count_corr[cmp][i] == 0:
                continue

            dcc = corr_day[cmp][i, :] / count_corr[cmp][i]
            dcc = dcc.astype("float32")

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

            tr = SACTrace(data=dcc, **header)
            filename = f"{pair}_{cmp}_{day}.{P.par['output_format']}"
            tr.write(os.path.join(P.par['corr_path'], cmp, pair, filename))
            day_pairs.append(pair)

    write_log(P.par['log_path'], day, day_pairs)
    ndays_proc -= 1
    if myrank == 0:
        print('~{} days left per core'.format(ndays_proc))

print(f"core {myrank} done")

if myrank == 0:
    shutil.copy(project_path, P.par['corr_path'])
