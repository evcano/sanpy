import numpy as np
import os
import shutil
import sys
from obspy import read


# PARAMETERS
# ==========
data_path = '/data/valeroe/cvc_correlations/events/observations'
data_format = 'sac'

output_format = 'semd'  # mseed, sac, semd

new_nt = 400
new_dt = 1.0
shift = -2.0  # specfem shift

rename_data = True
component = 'HXZ'

# DONT EDIT BELOW THIS LINE
# =========================
new_duration = (new_nt - 1) * new_dt
new_fs = 1. / new_dt
nyquist_fq = new_fs / 2.0

events = os.listdir(data_path)
events = [e for e in events
          if os.path.isdir(os.path.join(data_path, e))]

out_path = os.path.join('{}_reformated'.format(data_path))

if os.path.isdir(out_path):
    shutil.rmtree(out_path)

for event in events:
    event_path = os.path.join(data_path, event)
    out_path = os.path.join('{}_reformated'.format(data_path), event)

    os.makedirs(out_path, exist_ok=True)

    waveforms = os.listdir(event_path)

    for wf in waveforms:
        st = read(os.path.join(event_path, wf))
        tr = st[0]

        wf = wf.strip('.{}'.format(data_format))
        src, rec = wf.split('_')
        rec_net, rec_sta = rec.split('.')

        tr.stats.network = rec_net
        tr.stats.station = rec_sta

        if tr.stats.sampling_rate < nyquist_fq:
            print('New sampling rate is greater than current sampling rate')
            sys.exit()

        # antialiasing filter
        if tr.stats.sampling_rate > nyquist_fq:
            tr.taper(0.05)
            tr.filter('lowpass', freq=nyquist_fq, zerophase=True, corners=8)

        # downsample
        tr.interpolate(method='lanczos',
                       sampling_rate=new_fs,
                       a=1.0)

        tr.stats.sampling_rate = new_fs

        # if output format is semd, zero-pad trace start to account for
        # source function shift
        if output_format == 'semd':
            tr.trim(tr.stats.starttime + shift, tr.stats.endtime,
                    pad=True, fill_value=0.)

        # cut trace to new duration
        tr.trim(tr.stats.starttime, tr.stats.starttime + new_duration)

        # save trace
        if rename_data:
            out_name = '{}.{}.{}'.format(rec, component, output_format)
        else:
            out_name = wf

        out_name = os.path.join(out_path, out_name)

        if output_format == 'semd':
            times = tr.times()
            amp = tr.data

            out = np.zeros((times.shape[0], 2))
            out[:, 0] = times.copy() + shift
            out[:, 1] = amp.copy()

            np.savetxt(out_name, out, fmt='%.6f')
        else:
            tr.write(out_name, format=output_format)

    print('{} done'.format(event))
