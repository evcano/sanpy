import matplotlib.pyplot as plt
import numpy as np
import os
from obspy import read
from sanpy.base.project_functions import load_project
from convert_utm2geo import utm_geo


project_name = '../projects/sub_cvc_stacking.pkl'
data_format = 'sac'

utm_zone = 13

# slowness grid
smax = 0.7
nx = 100
ny = 100


# DONT EDIT BELOW THIS LINE
P = load_project(project_name)

sxaxis = np.linspace(-smax, smax, nx)
syaxis = np.linspace(-smax, smax, ny)

sdx = np.abs(sxaxis[1] - sxaxis[0])
sdy = np.abs(syaxis[1] - syaxis[0])

E = np.zeros((nx, ny))

for sta1 in P.stations.index:
    sta1_x, sta1_y = utm_geo(P.stations['lon'][sta1],
                             P.stations['lat'][sta1],
                             utm_zone, 2)

    for sta2 in P.stations.index:
        # skip autocorrelations
        if sta1 == sta2:
            continue

        # read and filter waveform
        st_path = os.path.join(P.par['corr_path'], 'all',
                               f'{sta1}_{sta2}.{data_format}')

        if os.path.isfile(st_path):
            st = read(st_path)
        else:
            st_path = os.path.join(P.par['corr_path'], 'all',
                                   f'{sta2}_{sta1}.{data_format}')

            if os.path.isfile(st_path):
                st = read(st_path)
                st[0].data = st[0].data[::-1]
            else:
                continue

        st.filter('bandpass', freqmin=0.03, freqmax=0.3, corners=4,
                  zerophase=True)

        # lag axis
        maxlag = ((st[0].stats.npts - 1) / 2) * st[0].stats.delta
        lags = np.linspace(-maxlag, maxlag, st[0].stats.npts)

        # compute location difference
        sta2_x, sta2_y = utm_geo(P.stations['lon'][sta2],
                                 P.stations['lat'][sta2],
                                 utm_zone, 2)

        xdif = (sta1_x - sta2_x) / 1000.0
        ydif = (sta1_y - sta2_y) / 1000.0


        # compute time shift and scores
        for i in range(0, nx):
            for j in range(0, ny):
                tshift = -sxaxis[i]*xdif - syaxis[j]*ydif
                idx = np.argmin(np.abs(lags - tshift))
                E[i, j] += st[0].data[idx]

plt.pcolor(sxaxis, syaxis, E.T)
plt.show()
np.save('beamforming_map', E)
