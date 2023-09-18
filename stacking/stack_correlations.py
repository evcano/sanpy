import numpy as np
import shutil
import sys
import os
from mpi4py import MPI
from obspy import read, Stream, UTCDateTime
from obspy.io.sac.sactrace import SACTrace

from sanpy.base.functions import (check_missing_logs,
                                  distribute_objects,
                                  write_log)

from sanpy.base.project_functions import load_project
from sanpy.util.correlation_branches import correlation_branches
from sanpy.util.find_outliers import modified_zscore


comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

project_path = sys.argv[1]
season = sys.argv[2]

P = load_project(project_path)

corr_path = os.path.join(P.par['corr_path'], season)
greens_path = os.path.join(P.par['greens_path'], season)
log_path = os.path.join(P.par['log_path'], season)

pairs_list = P.pairs_list
pending_pairs = check_missing_logs(log_path, pairs_list)

if len(pending_pairs) == 0:
    print("No pending pairs.")
    comm.Abort()

pairs_to_stack = distribute_objects(pending_pairs, nproc, myrank)
npairs_proc = len(pairs_to_stack)

if myrank == 0:
    for cmp in P.par["corr_cmpts"]:
        dir_ = os.path.join(corr_path, cmp)
        if not os.path.isdir(dir_):
            os.makedirs(dir_)

        dir_ = os.path.join(greens_path, cmp)
        if not os.path.isdir(dir_):
            os.makedirs(dir_)

    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    print('Each rank will stack ~{} pairs'.format(npairs_proc))

if season != "all":
    y1, m1, y2, m2 = season.split('_')
    season_start = UTCDateTime('{}-{}-01'.format(y1, m1))
    season_end = UTCDateTime('{}-{}-01'.format(y2, m2))

for pair in pairs_to_stack:
    sta1, sta2 = pair.split("_")
    no_daily_corr = {}
    for cmp in P.par["corr_cmpts"]:
        data_path = os.path.join(P.par['data_path'], cmp, pair)
        waveform_files = os.listdir(data_path)

        if not waveform_files:
            no_daily_corr[cmp] = 0
            print('No data.')
            continue

        if season == 'all':
            st = read(os.path.join(data_path, "*"), format=P.par['data_format'])
        else:
            st = Stream()
            for file_ in waveform_files:
                st_head = read(file_, format=P.par['data_format'], headonly=True)
                tr_start = st_head[0].stats.starttime
                tr_end = st_head[0].stats.endtime
                if tr_start >= season_start and tr_end < season_end:
                    st += read(file_, format=P.par['data_format'])

        if len(st) == 0:
            no_daily_corr[cmp] = 0
            print('No data.')
            continue

        if P.par['remove_outliers']:
            X = [np.max(np.abs(tr.data)) for tr in st]
            X = np.asarray(X)

            outliers, inliers = modified_zscore(X, thr=3.5)

            # stack at least 60% of daily correlations
            keep_these_corr = int(len(st) * 0.6)

            if len(inliers) < keep_these_corr:
                print("Stacking less than 60% of the correlations, change thr")
                comm.Abort()

            st_inliers = Stream()
            for j in inliers:
                st_inliers.append(st[j])
        else:
            st_inliers = st

        no_daily_corr[cmp] = len(st_inliers)

        st_inliers.stack(stack_type="linear")
        data = st_inliers[0].data.astype("float32")

        header = {
            "kstnm": sta2,
            "kcmpnm": cmp,
            "stla": P.stations[sta2]['lat'],
            "stlo": P.stations[sta2]['lon'],
            "stel": P.stations[sta2]['elv'],
            "kevnm": sta1,
            "evla": P.stations[sta1]['lat'],
            "evlo": P.stations[sta1]['lon'],
            "evdp": P.stations[sta1]['elv'],
            "lcalda": 1,
            "dist": P.pairs[pair]['dis'],
            "delta": P.par["dt"],
            "b": -P.par['maxlag'],
            "e": P.par['maxlag']
        }

        tr = SACTrace(data=data, **header)
        outfile = os.path.join(corr_path, cmp, f"{pair}_{cmp}.sac")
        tr.write(outfile)

        if P.par['compute_greens']:
            tr = tr.to_obspy_trace()
            _, _, sym = correlation_branches(tr, branch='all')
            sym.data = np.diff(sym.data, n=1) * -1.0
            outfile = os.path.join(greens_path, cmp, f"{pair}_{cmp}.sac")
            sym.write(outfile)

    write_log(log_path, pair, [x for x in no_daily_corr.values()])

    npairs_proc -= 1
    print(f"{myrank} has {npairs_proc} pairs to stack")

if myrank == 0:
    shutil.copy(project_path, P.par['corr_path'])
