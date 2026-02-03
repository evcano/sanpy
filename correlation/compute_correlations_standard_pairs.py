import matplotlib.pyplot as plt
import itertools
import glob
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

project_path = sys.argv[1]
P = load_project(project_path)

# distribute jobs
if myrank == 0:
    print("checking pending station pairs", flush=True)

    pairs_list_no_autocorr = [x for x in P.pairs_list if x.split("_")[0] != x.split("_")[1]]

    pending_pairs = check_missing_logs(log_path=P.par['log_path'],
                                       log_names=pairs_list_no_autocorr,
                                      )

    if len(pending_pairs) == 0:
        print('No more pairs to correlate', flush=True)
        comm.Abort()
else:
    pending_pairs = None

pending_pairs = comm.bcast(pending_pairs, root=0)

pairs_to_correlate = distribute_objects(pending_pairs, nproc, myrank)
npairs_proc = len(pairs_to_correlate)

if myrank == 0:
    print('Each process will correlate ~{} pairs'.format(npairs_proc), flush=True)

# define frequency domain taper
fqax = np.fft.rfftfreq(P.par['corr_nfft'], P.par['dt'])

if P.par['fqtaper']:
    fqcorners = P.par['fqtaper']       
    fqtaper = cosine_sac_taper(freqs=fqax, flimit=fqcorners)
    flatfq_idx = np.where((fqax > fqcorners[1]) & (fqax < fqcorners[2]))
else:
    fqtaper = np.ones(fqax.size)       
    flatfq_idx = np.arange(0, fqax.size)

# define some constants
lags = correlation_lags(P.par['corr_npts'], P.par['corr_npts'])
maxlag = int(P.par["maxlag"] / P.par['dt'])  # maxlag to store (in samples)
store_lags = np.where(np.abs(lags) <= maxlag)[0]

# read transient signals
if P.par['remove_tsignals']:
    tsignals = read_tsignals(P.par['tsig_path'], P.stations_list, P.par['data_cmpts'])

list_of_days = list(P.waveforms_paths_perday.keys())
list_of_days.sort()

# loop over pairs
for pair in pairs_to_correlate:
    s1, s2 = pair.split("_")

    if s1 == s2:
        print("skipping autocorrelation", flush=True)
        write_log(P.par['log_path'], pair, ['none'])
        npairs_proc -= 1
        continue

    net1, sta1 = s1.split(".")
    net2, sta2 = s2.split(".")

    # loop over days
    for day in list_of_days:
        day_obj = UTCDateTime(day)
        day2 = day.replace("-","")

        # read files
        waveforms_files1 = []
        waveforms_files2 = []

        for cmp in P.par['data_cmpts']:
            waveforms_files1 += glob.glob(os.path.join(P.par['data_path'], net1, sta1, f"*{cmp}*{day2}*"))
            waveforms_files2 += glob.glob(os.path.join(P.par['data_path'], net2, sta2, f"*{cmp}*{day2}*"))

        if waveforms_files1 and waveforms_files2:
            waveforms_files = waveforms_files1 + waveforms_files2
        else:
            continue

        st = Stream()
        for file_ in waveforms_files:
            st.extend(read(file_, format=P.par['data_format']))

        # declare array to store the correlations of the day
        corr_day = {}
        count_corr = {}
        for cmp in P.par["corr_cmpts"]:
            corr_day[cmp] = np.zeros(P.par['save_npts'])
            count_corr[cmp] = 0

        # slide a window over the data
        for st_win in st.slide(P.par['corr_dur'],
                               P.par['corr_dur']-P.par["corr_overlap"]):

            # check that there is data
            if not st_win:
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

                stations_win[cmp] = [f"{tr.stats.network}.{tr.stats.station}" for tr in st_win_cmp]

                # check there is data for the two stations
                if len(stations_win[cmp]) != 2:
                    continue

                # time normalization
                for tr in st_win_cmp:
                    tr = ram_normalization(tr, 0.5)

                st_win_cmp.detrend("linear")
                st_win_cmp.detrend("demean")
                st_win_cmp.taper(0.05)

                data[cmp] = np.asarray([tr.data for tr in st_win_cmp])
                data_fft[cmp] = np.fft.rfftn(data[cmp],
                                             s=[P.par["corr_nfft"]],
                                             axes=[1],
                                             norm="backward")

            # determine available correlation components
            avail_data_cmpts = list(data.keys())
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

            # compute noise correlations
            corr = {}
            pairs_win = {}

            for cmp in avail_corr_cmpts:
                if cmp in ["EE", "NN", "ZZ"]:
                    dcmp = cmp[0]
                    i1 = stations_win[dcmp].index(s1)
                    i2 = stations_win[dcmp].index(s2)

                    # linear cross-correlation of sta1 with sta2 as in equation 11 of Tromp et al. 2010
                    tmp_corr = data_fft[dcmp][i1, :] * np.conj(data_fft[dcmp][i2, :])
                    # convert to time domain, this results in [pos_lags, neg_lags]
                    tmp_corr = np.real(np.fft.irfft(tmp_corr, P.par['corr_nfft'], norm="backward"))
                    # switch second and first halves of corr to obtain [neg_lags, pos_lags]
                    tmp_corr = np.fft.fftshift(tmp_corr)
                    # eliminate effect of zero-padding
                    tmp_corr = my_centered(tmp_corr, len(lags))
                    # store lags of interest
                    tmp_corr = tmp_corr[store_lags]
                    corr[cmp] = tmp_corr                 

                elif cmp in ["RR", "TT"]:
                    i1_E = stations_win["E"].index(s1)
                    i1_N = stations_win["N"].index(s1)
                    i2_E = stations_win["E"].index(s2)
                    i2_N = stations_win["N"].index(s2)
                  
                    if cmp == "RR":
                        w1 = np.cos(np.deg2rad(P.pairs[pair]["az"]))
                        w2 = np.sin(np.deg2rad(P.pairs[pair]["az"]))
                    elif cmp == "TT":
                        w1 = -np.sin(np.deg2rad(P.pairs[pair]["az"]))
                        w2 = np.cos(np.deg2rad(P.pairs[pair]["az"]))
                  
                    fft_sta1 = w1 * data_fft["N"][i1_N,:] + w2 * data_fft["E"][i1_E,:]
                    fft_sta2 = w1 * data_fft["N"][i2_N,:] + w2 * data_fft["E"][i2_E,:]
                  
                    # linear cross-correlation of sta1 with sta2 as in equation 11 of Tromp et al. 2010
                    tmp_corr = fft_sta1 * np.conj(fft_sta2)
                    # convert to time domain, this results in [pos_lags, neg_lags]
                    tmp_corr = np.real(np.fft.irfft(tmp_corr, P.par['corr_nfft'], norm="backward"))
                    # switch second and first halves of corr to obtain [neg_lags, pos_lags]
                    tmp_corr = np.fft.fftshift(tmp_corr)
                    # eliminate effect of zero-padding
                    tmp_corr = my_centered(tmp_corr, len(lags))
                    # store lags of interest
                    tmp_corr = tmp_corr[store_lags]
                    corr[cmp] = tmp_corr                 

                # stack correlations
                corr[cmp] /= np.max(np.abs(corr[cmp]))  # normalize by max amp
                corr_day[cmp] += corr[cmp]
                count_corr[cmp] += 1

        # save correlations of the day
        for cmp in P.par["corr_cmpts"]:
            if count_corr[cmp] == 0:
                continue

            dcc = corr_day[cmp] / count_corr[cmp]
            dcc = dcc.astype("float32")

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
                "b": 0.0,
                }

            tr = SACTrace(data=dcc, **header)
            filename = f"{pair}_{cmp}_{day}.{P.par['output_format']}"
            tr.write(os.path.join(P.par['corr_path'], cmp, pair, filename))

    # write log once done with all days of the pair
    write_log(P.par['log_path'], pair, ['none'])
    npairs_proc -= 1

    if myrank == 0:
        print(f'{npairs_proc} pairs left for rank {myrank}', flush=True)

print(f"core {myrank} done", flush=True)

if myrank == 0:
    shutil.copy(project_path, P.par['corr_path'])
