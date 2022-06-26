import os
from obspy import read
from obspy.signal import PPSD


# NOTE: This script does not preprocess the data, i.e., it does not remove
# linear trend nor instrument response. Therefore, you must preprocess the
# data before using this script.

# NOTE: Waveform must be in displacement units; the output PSD is in
# acceleration.

# NOTE: PSDS PARAMETERS:
# - stats: trace information
# - metadata: station metadata
# - db_bins: defines the y-axis of the psdpdf (minval, step, maxval)
# - ppsd_length: trace slice duration to compute psd
# - overlap: amount of overlap between trace slices
# - special_handling: controls the instrument response removal and trace
# - differentiation (set to None; more info see source code)
# - period_limits: sets the x-axis limits (min and max periods for the psdpdf)
# - period_step_octaves: step of the period bins (x-axis)
# - period_smoothing_width_octaves: width of the period bins (x-axis)

# NOTE: Starting at period_limit[0] we take a step <period_step_octaves>, this
# is the center for the first period bin, which has a width of
# <period_smotthing_width_octaves>. We repeat this until period_limit[-1].


# PARAMETERS
# ==========
station_name = ''
data_path = ''
output_path = ''

# DONT EDIT BELOW THIS LINE
# =========================
# we assume the instrument response has already been removed
paz = {'sensitivity': 1.0}

waveform_files = os.listdir(data_path)

# create PPSD object
# set special_handling='ringlaser' to avoid instrument response removal
# and convertion of units to acceleration
st = read(os.path.join(data_path, waveform_files[0]))

ppsd = PPSD(stats=st[0].stats,
            metadata=paz,
            ppsd_length=3600.0,
            overlap=0.5,
            skip_on_gaps=False,
            db_bins=(-200, -25, 1.0),
            period_limits=None,
            period_step_octaves=0.125,
            period_smoothing_width_octaves=0.5,
            special_handling='ringlaser')

# add waveforms
for wf in waveform_files:
    st = read(os.path.join(data_path, wf))

    # convert displacement to acceleration
    st.differentiate(method='gradient')
    st.differentiate(method='gradient')

    ppsd.add(st)
    print(wf)

output_file = os.path.join(output_path, '{}.ppsd'.format(station_name))

ppsd.plot(period_lim=(1.0, 200.0))
ppsd.save_npz(output_file)
