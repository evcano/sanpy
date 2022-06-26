import matplotlib.pyplot as plt
import numpy as np
import os
from sanpy.base.project_functions import load_project


project_path = '../projects/cvc_correlation.pkl'
output_path = '.'

# DONT EDIT BELOW THIS LINE
# =========================
P = load_project(project_path)

stations = P.stations.index.to_list()
data_span = P.data_span

all_psd = []  # all psds

for day in data_span:
    day_psd = []

    for sta in stations:
        psd_path = os.path.join(P.par['psd_path'], sta,
                                '{}_{}.npy'.format(sta, day))

        if os.path.isfile(psd_path):
            psd = np.load(psd_path)
            day_psd.append(psd)

    if not day_psd:
        continue

    print('{} {} psd'.format(day, len(day_psd)))

    day_psd = np.array(day_psd)
    day_model_psd = np.median(day_psd, axis=0)

    all_psd.append(day_model_psd)

all_psd = np.array(all_psd)
model = np.median(all_psd, axis=0)

np.save(os.path.join(output_path, 'noise_source_spectrum'), model)

# plot model
freq = np.fft.rfftfreq(2*len(model)-1, d=P.par['dt'])
plt.plot(freq, model)
plt.show()
plt.close()
