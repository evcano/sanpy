import numpy as np
import shutil
import sys
import os
from obspy import read

from sanpy.base.project_functions import load_project
from sanpy.control.functions import filter_station_pairs


project_path = sys.argv[1]

thr_ndays = 90
thr_snr = 10.0
thr_cc = 0.8
thr_nseasons = 2

save_reversed_correlations = True

P = load_project(project_path)

# quality control
filtered_pairs = filter_station_pairs(P, thr_ndays, thr_snr,
                                      thr_cc, thr_nseasons)

filtered_pairs = np.array(filtered_pairs)

print('Total station pairs: {}'.format(len(P.pairs.index)))
print('Filtered station pairs: {}'.format(len(filtered_pairs)))

sta1 = []
sources = []

for pair in filtered_pairs:
    a, _ = pair.split('_')
    sta1.append(a)

sta1 = np.array(sta1)
sources = np.unique(sta1)

obs_path = os.path.join(P.par['data_path'], 'all')
output_path = os.path.join(P.par['output_path'], 'observations')

if os.path.isdir(output_path):
    shutil.rmtree(output_path)

for src in sources:
    event_path = os.path.join(output_path, src)
    os.makedirs(event_path, exist_ok=True)

    rec_idx = np.where(sta1 == src)[0]
    observations = filtered_pairs[rec_idx]

    for obs in observations:
        _, rec = obs.split('_')

        obs_name = '{}_{}.{}'.format(src, rec, P.par['data_format'])
        obs_file = os.path.join(obs_path, obs_name)

        if not os.path.isfile(obs_file):
            print('{} observation not found'.format(obs))
            continue

        if save_reversed_correlations:
            st = read(obs_file)
            st[0].data = st[0].data[::-1]
            st.write(os.path.join(event_path, obs_name))
        else:
            shutil.copy(obs_file, event_path)

thr_file = os.path.join(output_path, 'used_thresholds.txt')

with open(thr_file, 'w') as _f:
    _f.write('Minimum daily observations: {}\n'.format(thr_ndays))
    _f.write('Minimum SNR: {}\n'.format(thr_snr))
    _f.write('Minimum seasonal variation: {}\n'.format(thr_cc))
    _f.write('Minimum number of seasons: {}\n'.format(thr_nseasons))
