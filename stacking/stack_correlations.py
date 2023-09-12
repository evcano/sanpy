import numpy as np
import shutil
import sys
import os
from glob import glob
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

# distribute jobs
project_path = sys.argv[1]
P = load_project(project_path)

season = sys.argv[2]
if season != "all":
    y1, m1, y2, m2 = season.split('_')
    season_start = UTCDateTime('{}-{}-01'.format(y1, m1))
    season_end = UTCDateTime('{}-{}-01'.format(y2, m2))

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
    if not os.path.isdir(corr_path):
        os.makedirs(corr_path)

    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    if not os.path.isdir(greens_path):
        os.makedirs(greens_path)

    print('Each rank will stack ~{} pairs'.format(npairs_proc))

# do jobs
#TODO: it may be better to create one folder per corr component
for pair in pairs_to_stack:
    stpath = os.path.join(P.par['data_path'], pair)
    for cmp in P.par["corr_cmpts"]:
        corr_files_cmp = glob(os.path.join(stpath,f"*{cmp}*"))
        ncorr = 0
        if not corr_files_cmp:
            print('No data.')
            continue

        if season != 'all':
            st = Stream()
            for corrfile in corr_files_cmp:
                st_head = read(corrfile,
                               format=P.par['data_format'],
                               headonly=True)

                tr_start = st_head[0].stats.starttime
                tr_end = st_head[0].stats.endtime
                if tr_start >= season_start and tr_end < season_end:
                    st += read(corrfile, format=P.par['data_format'])

        elif season == 'all':
            st = read(os.path.join(stpath, f"*{cmp}*"),
                      format=P.par['data_format'])

        if len(st) == 0:
            print('No data.')
            continue

        # discard correlations with high amplitude
        if P.par['remove_outliers']:
            X = [np.max(np.abs(tr.data)) for tr in st]
            X = np.asarray(X)

            outliers, inliers = modified_zscore(X, thr=3.5)

            # stack at least 60% of daily correlations
            keep_these_corr = int(len(st) * 0.6)

            if len(inliers) < keep_these_corr:
                print('Less than 60% of {} interstation correlations were\
                       classified as outliers. Reduce the threshold.')
                comm.Abort()

            st_inliers = Stream()

            for j in inliers:
                st_inliers.append(st[j])
        else:
            st_inliers = st

        # stack data
        ncorr = len(st_inliers)
        st_inliers.stack(stack_type="linear")
        data = st_inliers[0].data.astype("float32")

        # save stack
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
            "delta": P.par["dt"],
            "b": -P.par['maxlag'],
            "e": P.par['maxlag']}

        tr = SACTrace(data=data, **header)
        outfile = os.path.join(corr_path, f"{pair}_{cmp}.sac")
        tr.write(outfile)

        # compute greens function
        if P.par['compute_greens']:
            tr = tr.to_obspy_trace()
            _, _, sym = correlation_branches(tr, branch='all')
            sym.data = np.diff(sym.data, n=1) * -1.0
            outfile = os.path.join(greens_path, f"{pair}_{cmp}.sac")
            sym.write(outfile)

    write_log(log_path, pair, [ncorr])
    npairs_proc -= 1
    if myrank == 0:
        print('~{} pairs left per core'.format(npairs_proc))

if myrank == 0:
    shutil.copy(project_path, P.par['corr_path'])
