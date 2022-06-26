import numpy as np
import shutil
import sys
import os
from mpi4py import MPI
from obspy import read
from scipy.signal import correlate
from scipy.signal.windows import tukey

from sanpy.base.functions import (check_missing_logs,
                                  distribute_objects,
                                  write_log)

from sanpy.base.project_functions import load_project
from sanpy.util.correlation_branches import correlation_branches
from sanpy.util.windows import max_envelope_window
from sanpy.util.snr import compute_snr


comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

# distribute jobs
project_path = sys.argv[1]

P = load_project(project_path)

pending_pairs = check_missing_logs(P.par['log_path'], P.pairs.index)

if len(pending_pairs) == 0:
    print('No pending_pairs')
    comm.Abort()

pairs_to_do = distribute_objects(pending_pairs, nproc, myrank)
npairs_proc = len(pairs_to_do)

if myrank == 0:
    print('~{} pairs per core'.format(npairs_proc))

fqmin = P.par['filter'][0]
fqmax = P.par['filter'][1]
corners = P.par['filter'][2]

# do jobs
for i, pair in enumerate(pairs_to_do):
    pair_log = []

    stpath = os.path.join(P.par['data_path'], 'all',
                          '{}.{}'.format(pair, P.par['data_format']))

    if not os.path.isfile(stpath):
        pair_log.append('-99. -99. -99. -99.')
        write_log(P.par['log_path'], pair, pair_log)
        print('No data')
        continue

    st_ref = read(stpath, format=P.par['data_format'])

    st_ref.taper(0.02)
    st_ref.filter('bandpass',
                  freqmin=fqmin,
                  freqmax=fqmax,
                  corners=corners,
                  zerophase=True)

    if P.par['data_type'] == 'correlation':
        pos, neg, sym = correlation_branches(st_ref[0], branch='all')

        signal_window = max_envelope_window(sym,
                                            minperiod=1./fqmax,
                                            maxperiod=1./fqmin,
                                            src_dist=sym.stats.sac.dist,
                                            vmin=P.par['vmin'])

        pos_snr = compute_snr(pos, signal_window, P.par['noise_dur'])
        neg_snr = compute_snr(neg, signal_window, P.par['noise_dur'])

        pair_log.append('{} 1.0 {} 1.0'.format(pos_snr, neg_snr))

        pos_signal_ref = pos.data[signal_window[0]:signal_window[1]+1]
        pos_signal_ref *= tukey(pos_signal_ref.size)

        neg_signal_ref = neg.data[signal_window[0]:signal_window[1]+1]
        neg_signal_ref *= tukey(neg_signal_ref.size)

    elif P.par['data_type'] == 'green':
        signal_window = max_envelope_window(st_ref[0],
                                            minperiod=1./fqmax,
                                            maxperiod=1./fqmin,
                                            src_dist=st_ref[0].stats.sac.dist,
                                            vmin=P.par['vmin'])

        snr = compute_snr(st_ref[0], signal_window, P.par['noise_dur'])

        pair_log.append('{} 1.0 -99. -99.'.format(snr))

        signal_ref = st_ref[0].data[signal_window[0]:signal_window[1]+1]
        signal_ref *= tukey(signal_ref.size)

    for season in P.par['seasons']:
        stpath = os.path.join(P.par['data_path'], season,
                              '{}.{}'.format(pair, P.par['data_format']))

        if not os.path.isfile(stpath):
            pair_log.append('-99. -99. -99. -99.')
            print('No data')
            continue

        st = read(stpath, format=P.par['data_format'])

        st.taper(0.02)
        st.filter('bandpass',
                  freqmin=fqmin,
                  freqmax=fqmax,
                  corners=corners,
                  zerophase=True)

        if P.par['data_type'] == 'correlation':
            pos, neg, sym = correlation_branches(st[0], branch='all')

            pos_snr = compute_snr(pos, signal_window, P.par['noise_dur'])
            neg_snr = compute_snr(neg, signal_window, P.par['noise_dur'])

            # correlate seasonal withh reference observation
            pos_signal_season = pos.data[signal_window[0]:signal_window[1]+1]
            pos_signal_season *= tukey(pos_signal_season.size)

            neg_signal_season = neg.data[signal_window[0]:signal_window[1]+1]
            neg_signal_season *= tukey(neg_signal_season.size)

            pos_corr = correlate(pos_signal_ref, pos_signal_season)
            pos_corr /= (np.linalg.norm(pos_signal_ref) *
                         np.linalg.norm(pos_signal_season))
            pos_cor = np.max(pos_corr)

            neg_corr = correlate(neg_signal_ref, neg_signal_season)
            neg_corr /= (np.linalg.norm(neg_signal_ref) *
                         np.linalg.norm(neg_signal_season))
            neg_cor = np.max(neg_corr)

            pair_log.append('{} {} {} {}'.format(pos_snr, pos_cor,
                                                 neg_snr, neg_cor))

        elif P.par['data_type'] == 'green':
            snr = compute_snr(st[0], signal_window, P.par['noise_dur'])

            signal_season = st[0].data[signal_window[0]:signal_window[1]+1]
            signal_season *= tukey(signal_season.size)

            corr = correlate(signal_ref, signal_season)
            corr /= (np.linalg.norm(signal_ref) *
                     np.linalg.norm(signal_season))
            cor = np.max(corr)

            pair_log.append('{} {} -99. -99.'.format(snr, cor))

    write_log(P.par['log_path'], pair, pair_log)

    if myrank == 0:
        print('~{} pairs left per core'.format(npairs_proc-1-i))

if myrank == 0:
    shutil.copy(project_path, P.par['output_path'])
