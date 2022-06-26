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

# distribute jobs
project_path = sys.argv[1]
season = sys.argv[2]

P = load_project(project_path)

corr_path = os.path.join(P.par['corr_path'], season)
greens_path = os.path.join(P.par['greens_path'], season)
log_path = os.path.join(P.par['log_path'], season)

pending_pairs = check_missing_logs(log_path, P.pairs.index)

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
for pair in pairs_to_stack:
    stpath = os.path.join(P.par['data_path'], pair)
    corr_files = os.listdir(stpath)

    if not corr_files:
        print('No data.')
        write_log(log_path, pair, [0])
        continue

    if season != 'all':
        y1, m1, y2, m2 = season.split('_')
        start = UTCDateTime('{}-{}-01'.format(y1, m1))
        end = UTCDateTime('{}-{}-01'.format(y2, m2))

        st = Stream()

        for corrfile in corr_files:
            corrfile = os.path.join(stpath, corrfile)

            st_head = read(corrfile, format=P.par['data_format'],
                           headonly=True)

            tr_start = st_head[0].stats.starttime
            tr_end = st_head[0].stats.endtime

            if tr_start >= start and tr_end < end:
                st += read(corrfile, format=P.par['data_format'])
    elif season == 'all':
        st = read(os.path.join(stpath, '*'), format=P.par['data_format'])

    if len(st) == 0:
        print('No data.')
        write_log(log_path, pair, [0])
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
        "delta": P.par["dt"],
        "b": -P.par['maxlag'],
        "e": P.par['maxlag']}

    tr = SACTrace(data=data, **header)
    outfile = os.path.join(corr_path, '{}.sac'.format(pair))
    tr.write(outfile)

    # compute greens function
    if P.par['compute_greens']:
        tr = tr.to_obspy_trace()
        _, _, sym = correlation_branches(tr, branch='all')
        sym.data = np.diff(sym.data, n=1) * -1.0
        outfile = os.path.join(greens_path, '{}.sac'.format(pair))
        sym.write(outfile)

    write_log(log_path, pair, [ncorr])

    npairs_proc -= 1

    if myrank == 0:
        print('~{} pairs left per core'.format(npairs_proc))

if myrank == 0:
    shutil.copy(project_path, P.par['corr_path'])
