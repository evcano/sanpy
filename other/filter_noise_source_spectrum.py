import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.fft as sf
from obspy.signal.filter import bandpass
from scipy.signal import correlation_lags, tukey


# PARAMETERS
# ==========
spectrum_path = './noise_source_spectrum.npy'
output_path = '.'

# information of traces used to compute the spectrum
tr_dur = 3600.0
dt = 0.5

# output information
output_nt = 11000  # source function npts for one branch
output_dt = 0.03  # source function dt

# filter
fqmin = 0.1
fqmax = 0.2
corners = 4

# DONT EDIT BELOW THIS LINE
# =========================
# read spectrum
spectrum = np.load(spectrum_path)

# get frequency axis
corr_npts = int((tr_dur / dt) + 1)
nfft = sf.next_fast_len(2 * corr_npts - 1)
freq = np.fft.rfftfreq(nfft, d=dt)

# taper spectrum ends to remove spikes
plt.plot(freq, spectrum, 'k')

taper = tukey(spectrum.size, 0.01)
spectrum = spectrum * taper

plt.plot(freq, taper, 'r')
plt.plot(freq, spectrum, 'b')
plt.show()

# interpolate spectrum to a new sampling rate
freq2 = np.fft.rfftfreq(output_nt, d=output_dt)
spectrum2 = np.interp(freq2, freq, spectrum, left=0.0, right=0.0)

# get source time function
t = correlation_lags(output_nt, output_nt, mode='full') * output_dt
x = np.real(np.fft.irfft(spectrum2, t.size, norm='backward'))
x = np.fft.fftshift(x)

# filter and normalize source time function
x = bandpass(x, freqmin=fqmin, freqmax=fqmax, corners=corners,
             zerophase=True, df=1.0/output_dt)

x /= np.max(x)

# save source function
output = np.array([t, x], dtype='float32')
output = output.T
np.savetxt(os.path.join(output_path, 'S_squared'),
           output, fmt=['%1.7e', '%1.7e'])

# figures
plt.plot(freq, spectrum, 'k')
plt.plot(freq2, spectrum2, 'b')
plt.show()

plt.plot(t, x)
plt.show()
