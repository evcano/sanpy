#import time
import h5py
import numpy as np
import os
import sys
from mpi4py import MPI
from obspy import read, UTCDateTime, Stream, Trace
from obspy.signal.invsim import cosine_sac_taper
from scipy.signal import windows 
from scipy.ndimage import uniform_filter1d, convolve1d
from sanpy.base.functions import (check_missing_logs,
                                  distribute_objects,
                                  write_log)

from sanpy.base.project_functions import load_project
from sanpy.correlation.functions import *

# SAME AS OTHER CODE, ONLY DIFFERENCES IS WE SAVE ALL CORRELATIONS

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

    pending_pairs = check_missing_logs(log_path=P.par['log_path'],
                                       log_names=P.pairs_list,
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

# set constants
required_cmpts = {"EE": {"E"}, "NN": {"N"}, "ZZ": {"Z"}, "RR": {"E","N"}, "TT": {"E","N"}}

maxlag_samp = int(P.par["maxlag"] / P.par['dt'])  # maxlag to store (in samples)
lags = correlation_lags(P.par['corr_npts'], P.par['corr_npts'])
store_lags = np.where(np.abs(lags) <= maxlag_samp)[0]

list_of_days = list(P.waveforms_paths_perday.keys())
list_of_days.sort()

ndays = len(list_of_days)
nwindows = 1 + (86400 - P.par['corr_dur']) / (P.par['corr_dur'] - P.par['corr_overlap'])
outshape = (nwindows*ndays, store_lags.size)

ram_wsize = int(P.par['ram_win'] / P.par['dt'])
if ram_wsize % 2 == 0:
    ram_wsize += 1

# define frequency domain taper
fqax = np.fft.rfftfreq(P.par['corr_nfft'], P.par['dt'])

if P.par['fqtaper']:
    fqcorners = P.par['fqtaper']       
    fqtaper = cosine_sac_taper(freqs=fqax, flimit=fqcorners)
    flatfq_idx = np.flatnonzero((fqax > fqcorners[1]) & (fqax < fqcorners[2]))
else:
    fqtaper = np.ones(fqax.size)       
    flatfq_idx = np.arange(0, fqax.size)

# define kernel to smooth spectra
spec_smooth_win = windows.hann(P.par['sw_win'])
spec_smooth_win /= spec_smooth_win.sum()

# loop over pairs
for pair in pairs_to_correlate:
    #t0 = time.perf_counter()

    # list all files of the involved stations
    s1, s2 = pair.split("_")
    net1, sta1 = s1.split(".")
    net2, sta2 = s2.split(".")

    if s1 != s2:
        waveforms_files1 = P.waveforms_paths_sta[s1]
        waveforms_files2 = P.waveforms_paths_sta[s2]
        waveforms_files = waveforms_files1 + waveforms_files2
    else:
        waveforms_files = P.waveforms_paths_sta[s1]

    # set constants
    pair_az = np.deg2rad(P.pairs[pair]["az"])
    cosaz = np.cos(pair_az)
    sinaz = np.sin(pair_az)

    # set up HDF5 file
    outfile = os.path.join(P.par['corr_path'], f'{pair}.h5')
    h5file = h5py.File(outfile, 'w')

    h5file.attrs["delta"] = P.par['dt']
    h5file.attrs["b"] = P.par['maxlag']
    h5file.attrs["lcalda"] = 1
    h5file.attrs["kstnm"] = s2
    h5file.attrs["stla"] = P.stations[s2]['lat']
    h5file.attrs["stlo"] = P.stations[s2]['lon']
    h5file.attrs["stel"] = P.stations[s2]['elv']
    h5file.attrs["kevnm"] = s1
    h5file.attrs["evla"] = P.stations[s1]['lat']
    h5file.attrs["evlo"] = P.stations[s1]['lon']
    h5file.attrs["evdp"] = P.stations[s1]['elv']
    h5file.attrs["dist"] = P.pairs[pair]['dis']

    DSETS = {}

    for cmp in P.par["corr_cmpts"]:
        dset_data = h5file.create_dataset(
            cmp,
            shape=outshape,
            maxshape=(None, outshape[1]),
            dtype=np.float32,
        )

        dset_dates = h5file.create_dataset(
            f"{cmp}_dates",
            shape=(outshape[0],),
            maxshape=(None,),
            dtype=np.float64,
        )

        DSETS[cmp] = {'data': dset_data, 'dates': dset_dates, 'c': 0}

    for day in list_of_days:
        # read all data of the day (including all components)
        day2 = day.replace("-","")
        day_waveforms_files = [os.path.join(P.par['data_path'], f) for f in waveforms_files if day2 in f]

        if day_waveforms_files:
            st = Stream()
            for file_ in day_waveforms_files:
                st += read(file_, format=P.par['data_format'])
        else:
            continue

        # prepare lists for the day
        corr_fft_day = []
        corr_cmp_day = []
        corr_date_day = []

        # slide a window over the data
        for st_win in st.slide(P.par['corr_dur'], P.par['corr_dur']-P.par["corr_overlap"]):
            # remove traces with time gaps and check available data components
            st_win = Stream([tr for tr in st_win if tr.stats.npts == P.par['corr_npts']])


            sta_cmpts = {sta1: [], sta2: []}
            for tr in st_win:
                sta_cmpts[tr.stats.station].append(tr.stats.component)

            # obtain data components available for both stations
            avail_data_cmpts = [
                cmp for cmp in P.par["data_cmpts"]
                if cmp in sta_cmpts[sta1] and cmp in sta_cmpts[sta2]
            ]
            avail_data_cmpts = set(avail_data_cmpts)

            # check available correlation components
            avail_corr_cmpts = [
                cmp for cmp in P.par["corr_cmpts"]
                if required_cmpts[cmp] <= avail_data_cmpts
            ]

            if not avail_corr_cmpts:
                continue

            window_date = st_win[0].stats.starttime.timestamp

            # preprocessing
            st_win.detrend("linear")
            st_win.detrend("demean")
            st_win.taper(0.05)

            # arrange data into a matrix
            DATA = []
            MAP = {}
            for i, tr in enumerate(st_win):
                DATA.append(tr.data)
                MAP[f"{tr.stats.network}.{tr.stats.station}.{tr.stats.component}"] = i
            DATA = np.asarray(DATA, dtype=np.float32)

            # running-absolute-mean normalization
            W = uniform_filter1d(np.abs(DATA), size=ram_wsize, axis=1, mode='reflect')
            W[W < 1e-10] = 1e-10
            DATA /= W

            # compute fft
            DATA_FFT = np.fft.rfftn(DATA, s=[P.par["corr_nfft"]], axes=[1], norm="backward")

            # frequency taper
            DATA_FFT *= fqtaper

            # spectral whitening
            AMP = np.abs(DATA_FFT)
            norm_spec = convolve1d(AMP, weights=spec_smooth_win, axis=1, mode='reflect')
            DATA_FFT /= (norm_spec + 1e-10)

            # separate amplitude and phase
            AMP = np.abs(DATA_FFT)
            PHASE = DATA_FFT / np.maximum(AMP, 1e-10)

            # detect spikes
            TMP = AMP[:, flatfq_idx]
            imin = np.percentile(TMP, 5, axis=1)
            imax = np.percentile(TMP, 95, axis=1)
            TMP = np.where(
                (TMP >= imin[:, None]) & (TMP <= imax[:, None]),
                TMP,
                np.nan
            )
            rms = np.nanstd(TMP, axis=1)

            # clip magnitude, preserve phase
            AMP = np.minimum(AMP, rms[:,None])
            DATA_FFT = AMP * PHASE

            # rotate components
            if "RR" in avail_corr_cmpts or "TT" in avail_corr_cmpts:
                i1_E = MAP[f"{s1}.E"]
                i1_N = MAP[f"{s1}.N"]

                i2_E = MAP[f"{s2}.E"]
                i2_N = MAP[f"{s2}.N"]

                R1 = cosaz * DATA_FFT[i1_N,:] + sinaz * DATA_FFT[i1_E,:]
                R2 = cosaz * DATA_FFT[i2_N,:] + sinaz * DATA_FFT[i2_E,:]

                T1 = -sinaz * DATA_FFT[i1_N,:] + cosaz * DATA_FFT[i1_E,:]
                T2 = -sinaz * DATA_FFT[i2_N,:] + cosaz * DATA_FFT[i2_E,:]

            # compute noise correlations
            for cmp in avail_corr_cmpts:
                if cmp in ("EE", "NN", "ZZ"):
                    fft_sta1 = DATA_FFT[MAP[f"{s1}.{cmp[0]}"],:]
                    fft_sta2 = DATA_FFT[MAP[f"{s2}.{cmp[0]}"],:]
                elif cmp == "RR":
                    fft_sta1 = R1
                    fft_sta2 = R2
                elif cmp == "TT":
                    fft_sta1 = T1
                    fft_sta2 = T2

                # linear cross-correlation of sta1 with sta2 as in equation 11 of Tromp et al. 2010
                corr_fft = fft_sta1 * np.conj(fft_sta2)

                # append
                corr_fft_day.append(corr_fft)
                corr_cmp_day.append(cmp)
                corr_date_day.append(window_date)

        if not corr_fft_day:
            continue

        # arrange ffts in one matrix
        CORR_FFT = np.array(corr_fft_day)

        # convert to time domain, this results in [pos_lags, neg_lags]
        CORR = np.real(np.fft.irfft(CORR_FFT, n=P.par['corr_nfft'], axis=1, norm="backward"))

        # switch second and first halves of corr to obtain [neg_lags, pos_lags]
        CORR = np.fft.fftshift(CORR, axes=1)

        # eliminate effect of zero-padding conducted to accelerate fft
        CORR = my_centered2d(CORR, lags.size)

        # store lags of interest
        CORR = CORR[:, store_lags]

        # pass correlations of the day to HDF5
        corr_cmp_day = np.array(corr_cmp_day)
        corr_date_day = np.array(corr_date_day)
        cmp_idx = {cmp: np.where(corr_cmp_day == cmp)[0] for cmp in P.par["corr_cmpts"]}

        for cmp in P.par["corr_cmpts"]:
            idx = cmp_idx[cmp]
            CORR_out = CORR[idx]
            dates_out = corr_date_day[idx]

            c = DSETS[cmp]['c']            
            m = idx.size

            DSETS[cmp]['data'][c:c+m] = CORR_out
            DSETS[cmp]['dates'][c:c+m] = dates_out
            DSETS[cmp]['c'] += m

    # resize HDF5 datasets and write file
    for cmp in P.par["corr_cmpts"]:
        c = DSETS[cmp]['c']
        DSETS[cmp]['data'].resize(c, axis=0)
        DSETS[cmp]['dates'].resize(c, axis=0)

    h5file.flush()
    h5file.close()

    # write log once done with all days of the pair
    write_log(P.par['log_path'], pair, ['none'])
    npairs_proc -= 1

    print(f'{pair} done; {npairs_proc} pairs left for rank {myrank}', flush=True)

    #t1 = time.perf_counter()
    #print("elapsed:", t1 - t0, "seconds", flush=True)

print(f"core {myrank} done", flush=True)
