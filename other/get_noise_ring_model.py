import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


beamform = './beamforming_map.npy'

amp_factor = 100.0E+20

# slowness grid
smax = 0.7
nx = 100
ny = 100

# integrate beamforming map between this slowness range
thr1 = 0.25
thr2 = 0.4

# DONT EDIT BELOW THIS LINE
# =========================
sxaxis = np.linspace(-smax, smax, nx)
syaxis = np.linspace(-smax, smax, ny)
E = np.load(beamform)

# flat arrays
X = np.array([])
Y = np.array([])
Z = np.array([])

for i in range(0, nx):
    for j in range(0, ny):
        X = np.append(X, sxaxis[i])
        Y = np.append(Y, syaxis[j])
        Z = np.append(Z, E[i, j])

# generate a regular grid in polar coordinates
rmax = np.sqrt(smax**2 + smax**2)
dr = rmax / nx
ax_r = np.arange(0.0, rmax + dr, dr)

dtheta = (2*np.pi) / 360.0
ax_theta = np.arange(0.0, 2*np.pi + dtheta, dtheta)

R, T = np.meshgrid(ax_r, ax_theta)
Xnew = R * np.cos(T)
Ynew = R * np.sin(T)

# interpolate beamforming map into the regular grid
points = np.zeros((X.size, 2))
points[:, 0] = np.squeeze(X)
points[:, 1] = np.squeeze(Y)

Znew = griddata(points, Z, (Xnew, Ynew), method='linear')

# energy
Znew = Znew ** 2

# mute
Znew_bak = Znew.copy()
idx = np.where(R < thr1)
Znew[idx] = 0.0
idx = np.where(R > thr2)
Znew[idx] = 0.0

# integrate
r = np.zeros(ax_r.size)
r[1:] = 0.5 * (ax_r[0:-1] + ax_r[1:])
da = r * dr * dtheta

noise_ring = np.zeros(ax_theta.size)

for i, t in enumerate(ax_theta):
    idx = np.where(T == t)
    noise_ring[i] = np.sum(Znew[idx] * da)

# plot figures
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
f = ax.scatter(T, R, c=Znew_bak)
plt.title('Beamforming map')
plt.colorbar(f)
plt.show()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
f = ax.scatter(T, R, c=Znew)
plt.title('Muted beamforming map')
plt.colorbar(f)
plt.show()

plt.plot(np.rad2deg(ax_theta), noise_ring, 'o')
plt.title('Azimuthal noise distribution')
plt.show()

# save file
noise_ring *= amp_factor / np.max(noise_ring)
ax_theta = np.rad2deg(ax_theta)

out = np.array([ax_theta, noise_ring]).T
np.savetxt('noise_ring_model.txt', out)
