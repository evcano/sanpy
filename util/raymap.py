import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os


def plot_ray_coverage(P, latmin, latmax, lonmin, lonmax, pairs2invert,
                      rayvalue=None, showrays=True):

    # set map projection and limits
    mercator = ccrs.PlateCarree()
    extent = [lonmin, lonmax, latmin, latmax]

    # plot map
    ax = plt.axes(projection=mercator)
    ax.set_extent(extent)

    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      color='black', alpha=0.2, linewidth=0.25)

    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 7.5}
    gl.ylabel_style = {'size': 7.5}

    # plot stations
    for sta in P.stations_list:
        lon = P.stations[sta]["lon"]
        lat = P.stations[sta]["lat"]

        plt.plot(lon, lat, marker='v', markersize=3,
                 color="r", markerfacecolor="r", transform=mercator)

        plt.text(lon, lat, sta, fontsize=8.0, transform=mercator)

    # rays
    if rayvalue.any():
        rvmin = 0.5 #max(np.floor(np.nanmin(rayvalue)), 1.0)
        rvmax = 4.5#
        norm = matplotlib.colors.Normalize(vmin=rvmin,vmax=rvmax,clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
        plt.colorbar(mappable=mapper, ax=ax)

    dont_plot_src = []

    for i, pair in enumerate(pairs2invert):
        sta1, sta2 = pair.split("_")

        sta1lon = P.stations[sta1]["lon"]
        sta1lat = P.stations[sta1]["lat"]

        sta2lon = P.stations[sta2]["lon"]
        sta2lat = P.stations[sta2]["lat"]

        if showrays:
            col = mapper.to_rgba(rayvalue[i])
            alpha = 1.
            if np.isnan(rayvalue[i]):
                col = "w"
                alpha = 0.0

            plt.plot([sta1lon, sta2lon], [sta1lat, sta2lat],
                     color=col, alpha=alpha, linewidth=0.9,
                     transform=ccrs.Geodetic())

#    ax.add_feature(cfeature.LAND)
#    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)

    return ax
