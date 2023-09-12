import os
import shutil
import sys
from mpi4py import MPI
from obspy import read, read_inventory

from sanpy.base.functions import distribute_objects
from sanpy.base.project_functions import load_project

from sanpy.preprocessing.functions import (check_sample_aligment,
                                           fill_gaps,
                                           get_pending_waveforms,
                                           preprocess)


"""
this script assumes mseed files contaning waveforms from only one channel
of one station during one day

data must be as: data_dir/network/station/waveforms
each process reads and process only 1 one-day waveform at the time

all proccessed waveforms are saved on the same folder, ignoring net and sta

"""


comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

# distribute jobs
project_path = sys.argv[1]

P = load_project(project_path)

pending_waveforms = get_pending_waveforms(P.par['output_path'],
                                          P.waveforms_paths)

if len(pending_waveforms) == 0:
    print("No more waveforms to proccess.")
    comm.Abort()

waveforms_to_process = distribute_objects(pending_waveforms, nproc, myrank)
nwaveforms_proc = len(waveforms_to_process)

if myrank == 0:
    print(f"To do ~{nwaveforms_proc} waveforms per core")

# do jobs
sta_previous = "dummy"

for i, waveform_path in enumerate(waveforms_to_process):
    net, sta, waveform_name = waveform_path.split("/")
    data_format = P.par['data_format'][net]

    # read inventory
    if sta != sta_previous:
        inv_name = f"{net}.{sta}.xml"
        inv_file = os.path.join(P.par['metadata_path'], inv_name)
        inv = read_inventory(inv_file)

    # read waveform
    tmp = os.path.join(P.par['data_path'], waveform_path)
    st = read(tmp, format=data_format)

    # preproccess waveform (inplace)
    check_sample_aligment(st)
    fill_gaps(st, P.par)
    preprocess(st, P.par, inv)

    # ensure single precision float
    for tr in st:
        tr.data = tr.data.astype("float32")

    # save waveform
    waveform_name_noext = waveform_name[:-len(data_format)-1]
    out_file = f"{waveform_name_noext}.{P.par['output_format']}"
    out_path = os.path.join(P.par['output_path'], net, sta, out_file)
    st.write(out_path, format=P.par["output_format"])

    sta_previous = sta

    nwaveforms_proc -= 1

    if myrank == 0:
        print(f"~{nwaveforms_proc} waveforms left per core")

if myrank == 0:
    shutil.copy(project_path, P.par['output_path'])
