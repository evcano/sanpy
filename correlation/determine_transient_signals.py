import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
from glob import glob
from mpi4py import MPI
from scipy.stats import zscore
from obspy import UTCDateTime
from sanpy.base.functions import distribute_objects
from sanpy.base.project_functions import load_project


comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

project_path = sys.argv[1]
P = load_project(project_path)

stations_list = P.stations_list
stations_to_process = distribute_objects(stations_list, nproc, myrank)

# station loop
for sta in stations_to_process:
    fname = os.path.join(P.par["tsig_path"], f"{sta}_max_amps")
    if not os.path.isfile(fname):
        continue
    with open(fname,"rb",-1) as _file:
        max_amps_dic = pickle.load(_file)

    for cha in max_amps_dic.keys():
        sta_outl_stimes = []

        for band in max_amps_dic[cha].keys():
            max_amps = max_amps_dic[cha][band]

            stimes = np.array([x[0] for x in max_amps],dtype=str)
            amps = np.array([x[1] for x in max_amps],dtype=float)    # double
            outl_stimes = []

            # detect outliers based on 95 percentile
            p = np.percentile(amps, P.par["tsig_percentile"])
            idx = np.where(amps > p)[0].tolist()
            outl_stimes.extend(stimes[idx])

            # keep inliers 
            idx = np.where(amps <= p)[0].tolist()
            stimes2 = stimes[idx]
            amps2 = amps[idx]

            # detect outliers based on zscore
            z = zscore(amps2)
            idx = np.where(abs(z) > P.par["tsig_zscore"])[0]
            outl_stimes.extend(stimes2[idx])

            # store outliers in this frequency band
            sta_outl_stimes.extend(outl_stimes)

        # eliminate duplicated outliers
        sta_outl_stimes = list(set(sta_outl_stimes))
        sta_outl_stimes.sort()

        # get end times of transient signals
        sta_outl_etimes = [UTCDateTime(x) + P.par["corr_dur"]
                           for x in sta_outl_stimes]

        sta_outl_etimes = [x.format_iris_web_service() for x in sta_outl_etimes]

        # write channel transient signals
        outfile = os.path.join(P.par["tsig_path"], f"{sta}.{cha}_transient_signals.dat")

        with open(outfile, "w") as _file:
            for i in range(len(sta_outl_stimes)):
                _file.write(
                    f"{sta_outl_stimes[i]} {sta_outl_etimes[i]} \n"
                )

        # print info
        nwin = len(stimes)
        ntrans = len(sta_outl_stimes)
        print(f"station : {sta}.{cha}")
        print(f"outliers: {ntrans}/{nwin} ({ntrans/nwin}%)\n")
