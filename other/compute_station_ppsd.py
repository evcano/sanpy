import os
from obspy import read, read_inventory
from obspy.signal import PPSD


# NOTE: PPSD removes linear trend, instrument response, and converts units
# from velocity to acceleration. It does NOT remove the mean nor applies
# a filter.

# NOTE: Waveform must be in velocity units; the output PSD is in acceleration.

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

# NOTE: starting at period_limit[0] we take a step <period_step_octaves>, this
# is the center for the first period bin, which has a width of
# <period_smotthing_width_octaves>. We repeat this until period_limit[-1].


# PARAMETERS
# ==========
station_name = ''
data_path = ''
metadata_path = ''
output_path = ''


# DONT EDIT BELOW THIS LINE
# =========================
inv = read_inventory(metadata_path)
waveform_files = os.listdir(data_path)


# create PPSD object
st = read(os.path.join(data_path, waveform_files[0]))

ppsd = PPSD(stats=st[0].stats,
            metadata=inv,
            ppsd_length=3600.0,
            overlap=0.5,
            skip_on_gaps=False,
            db_bins=(-200, -25, 1.0),
            period_limits=None,
            period_step_octaves=0.125,
            period_smoothing_width_octaves=0.5,
            special_handling=None)

# add waveforms
for wf in waveform_files:
    st = read(os.path.join(data_path, wf))
    st.detrend(type='demean')
    ppsd.add(st)
    print(wf)

output_file = os.path.join(output_path, '{}.ppsd'.format(station_name))

ppsd.plot(period_lim=(1.0, 200.0))
ppsd.save_npz(output_file)
