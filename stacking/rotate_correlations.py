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
corr_path = os.path.join(P.par['corr_path'], season)
greens_path = os.path.join(P.par['greens_path'], season)
#log_path = os.path.join(P.par['log_path'], season)

pairs_list = P.pairs_list
#pending_pairs = check_missing_logs(log_path, pairs_list)
pending_pairs = pairs_list

if len(pending_pairs) == 0:
    print("No pending pairs.")
    comm.Abort()

pairs_to_rotate = distribute_objects(pending_pairs, nproc, myrank)
npairs_proc = len(pairs_to_rotate)

if myrank == 0:
    print('Each rank will stack ~{} pairs'.format(npairs_proc))

dfmt = P.par["data_format"]

# do jobs
#TODO: it may be better to create one folder per corr component
for pair in pairs_to_rotate:
    ncorr = 0
    s1, s2 = pair.split("_")

    # read EE, NN, NE components
    tr_dic = {}
    try:
        for cmp in ["EE","NN","NE","EN"]:
            if cmp  == "EN":
                # use correlation conjugate symmetry
                st = read(os.path.join(corr_path,
                                       f"{s2}_{s1}_NE.{dfmt}"))
                tr = st[0]
                tr.data = tr.data[::-1]
                tr_dic[cmp] = tr.data
            else:
                st = read(os.path.join(corr_path,
                                       f"{s1}_{s2}_{cmp}.{dfmt}"))
                tr = st[0]
                tr_dic[cmp] = tr.data

    except Exception:
        print("one of the EE,NN,NE,EN components is missing; skipping")
        continue

    # rotate traces
    print(s1,s2)
    az = np.deg2rad(P.pairs[f"{s1}_{s2}"]["az"])
    baz = np.deg2rad(P.pairs[f"{s1}_{s2}"]["baz"])

    w1 = -np.cos(az)*np.cos(baz)
    w2 = np.cos(az)*np.sin(baz)
    w3 = -np.sin(az)*np.sin(baz)
    w4 = np.sin(az)*np.sin(baz)
    w5 = -np.sin(az)*np.cos(baz)
    w6 = -np.cos(az)*np.sin(baz)
    w7 = np.sin(az)*np.sin(baz)

    TT = w1*tr_dic["EE"]+w2*tr_dic["EN"]+w3*tr_dic["NN"]+w4*tr_dic["NE"]
    RR = w3*tr_dic["EE"]+w5*tr_dic["EN"]+w1*tr_dic["NN"]+w6*tr_dic["NE"]
    TR = w6*tr_dic["EE"]+w1*tr_dic["EN"]+w4*tr_dic["NN"]+w7*tr_dic["NE"]
    RT = w5*tr_dic["EE"]+w7*tr_dic["EN"]+w2*tr_dic["NN"]+w1*tr_dic["NE"]


    # save rotated traces
    header = {
        "kstnm": s2,
        "kcmpnm": None,
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

    header["kcmpnm"] = "TT"
    tr = SACTrace(data=TT, **header)
    outfile = os.path.join(corr_path, f"{pair}_TT.sac")
    tr.write(outfile)

    header["kcmpnm"] = "RR"
    tr = SACTrace(data=RR, **header)
    outfile = os.path.join(corr_path, f"{pair}_RR.sac")
    tr.write(outfile)

    header["kcmpnm"] = "TR"
    tr = SACTrace(data=TR, **header)
    outfile = os.path.join(corr_path, f"{pair}_TR.sac")
    tr.write(outfile)

    header["kcmpnm"] = "RT"
    tr = SACTrace(data=RT, **header)
    outfile = os.path.join(corr_path, f"{pair}_RT.sac")
    tr.write(outfile)

    npairs_proc -= 1
    if myrank == 0:
        print('~{} pairs left per core'.format(npairs_proc))

if myrank == 0:
    shutil.copy(project_path, P.par['corr_path'])
