import fnmatch
import numpy as np
import os
import sys
import pickle
from glob import glob
from mpi4py import MPI
from obspy import read
from sanpy.base.functions import distribute_objects
from sanpy.base.project_functions import load_project


def get_pending_stations(P):
    pending_stations = []
    for sta in P.stations_list:
        tmp = os.path.join(P.par["tsig_path"], f"{sta}_max_amps")
        if not os.path.isfile(tmp):
            pending_stations.append(sta)
    return pending_stations


comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

project_path = sys.argv[1]
P = load_project(project_path)

pending_stations = get_pending_stations(P)
stations_to_process = distribute_objects(pending_stations, nproc, myrank)

# loop over stations
for sta in stations_to_process:
    maxamp_dic = {}
    # loop over component
    for cmp in P.par["data_cmpts"]:
        wf_files = [x for x in P.waveforms_paths
                    if fnmatch.fnmatch(x, f"*{sta}.??{cmp}*")]

        maxamp_dic[cmp] = {}
        # prepare output dictionary
        for band in P.par["tsig_fqbands"]:
            fqmin = band[0]
            fqmax = band[1]
            bcode = f"{fqmin}-{fqmax}"
            maxamp_dic[cmp][bcode] = []
        # loop over component waveforms
        for file_ in wf_files:
            file_ = os.path.join(P.par["data_path"], file_)
            st = read(file_, format=P.par['data_format'])
            st.detrend("linear")
            st.detrend("demean")
            st.taper(0.05,type="hann")
            # loop over frequency band
            for band in P.par["tsig_fqbands"]:
                fqmin = band[0]
                fqmax = band[1]
                bcode = f"{fqmin}-{fqmax}"

                st2 = st.copy()
                st2.filter("bandpass",
                           freqmin=fqmin,
                           freqmax=fqmax,
                           corners=4,
                           zerophase=True,
                          )
                # slide window
                for tr in st2:
                    for tr_win in tr.slide(P.par["corr_dur"],
                                           P.par["corr_dur"]-P.par["corr_overlap"]):

                        stime = tr_win.stats.starttime.format_iris_web_service()
                        maxamp = np.max(np.abs(tr_win.data))
                        maxamp_dic[cmp][bcode].append((stime,maxamp))

            print(file_, flush=True)

    # write station results
    outfile = os.path.join(P.par["tsig_path"], f"{sta}_max_amps")
    with open(outfile,"wb") as _file:
        pickle.dump(maxamp_dic, _file, -1)

    print(f"{sta}.{cmp} done", flush=True)
