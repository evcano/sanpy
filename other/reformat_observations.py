import numpy as np
import os
import shutil
import sys
from obspy import read


# PARAMETERS
# ==========
data_path = '/data/valeroe/sub_cvc_correlations/events/observations'
data_format = 'sac'
correlation = True

new_duration = 89.5  # this equals maximum lag if correlation=True
new_dt = None
shift = 0.0  # specfem shift

rename_data = True
output_format = 'semd'  # mseed, sac, semd
output_component = 'BXZ'
output_lags = False  # output lags instead of times starting from 0


# DONT EDIT BELOW THIS LINE
# =========================
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

        # resample data
        if new_dt:
            new_fs = 1. / new_dt
            nyquist_fq = new_fs / 2.0

            if tr.stats.sampling_rate < nyquist_fq:
                print('New sampling rate is larger than current sampling rate')
                sys.exit()

            if tr.stats.sampling_rate > nyquist_fq:
                tr.taper(0.05)
                tr.filter('lowpass', freq=nyquist_fq, zerophase=True,
                          corners=8)

            tr.interpolate(method='lanczos',
                           sampling_rate=new_fs,
                           a=1.0)

            tr.stats.sampling_rate = new_fs
            tr.stats.delta = new_dt

        # cut data
        if new_duration:
            if correlation:
                maxlag = ((tr.stats.npts - 1) / 2) * tr.stats.delta
                tr.trim(tr.stats.starttime + maxlag - new_duration,
                        tr.stats.starttime + maxlag + new_duration)
            else:
                tr.trim(tr.stats.starttime, tr.stats.starttime + new_duration)

        # if output format is semd, zero-pad data start to account for
        # source function shift
        if output_format == 'semd' and shift != 0.0:
            tr.trim(tr.stats.starttime + shift, tr.stats.endtime,
                    pad=True, fill_value=0.)

        # save data
        if rename_data:
            out_name = '{}.{}.{}'.format(rec, output_component, output_format)
        else:
            out_name = wf

        out_name = os.path.join(out_path, out_name)

        if output_format == 'semd':
            times = tr.times()

            if output_lags:
                times -= new_duration

            amp = tr.data

            out = np.zeros((times.shape[0], 2))
            out[:, 0] = times.copy() + shift
            out[:, 1] = amp.copy()

            np.savetxt(out_name, out, fmt='%.6f')
        else:
            tr.write(out_name, format=output_format)

    print('{} done'.format(event))
