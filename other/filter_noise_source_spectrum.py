import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.fft as sf
from obspy.signal.invsim import cosine_sac_taper
from scipy.signal import correlation_lags


# PARAMETERS
# ==========
spectrum_path = './noise_source_spectrum.npy'
output_path = '.'

# information of traces used to compute the spectrum
tr_dur = 3600.0
dt = 0.5

# output information
flimit = (0.006, 0.03, 0.33, 0.594)  # filter corners
output_nt = 3000  # source function npts for one branch
output_dt = 0.03  # source function dt

# DONT EDIT BELOW THIS LINE
# =========================
corr_npts = int((tr_dur / dt) + 1)
nfft = sf.next_fast_len(2 * corr_npts - 1)

spectrum = np.load(spectrum_path)
freq = np.fft.rfftfreq(nfft, d=dt)

# filter spectrum
taper = cosine_sac_taper(freq, flimit)
spectrum = spectrum * taper
spectrum /= np.max(spectrum)
output = np.array([freq, spectrum])

# save spectrum
np.save(os.path.join(output_path, 'filtered_noise_source_spectrum'), output)

# get source function from the spectrum
freq2 = np.fft.rfftfreq(output_nt, d=output_dt)
spectrum2 = np.interp(freq2, freq, spectrum)
spectrum2 /= np.max(spectrum2)

t = correlation_lags(output_nt, output_nt, mode='full') * output_dt
x = np.real(np.fft.irfft(spectrum2, t.size, norm='backward'))
x = np.fft.fftshift(x)

output = np.array([t, x], dtype='float32')
output = output.T

# save source function
np.savetxt(os.path.join(output_path, 'S_squared'),
           output, fmt=['%1.7e', '%1.7e'])

# figures
plt.plot(freq, spectrum, 'k')
plt.plot(freq2, spectrum2, 'b')
plt.show()

plt.plot(t, x)
plt.show()
