import matplotlib.pyplot as plt
import os
from sanpy.util.plot import plot_daily_correlations
from glob import glob


# PARAMETERS
# =========
data_path = ""
data_format = 'sac'
bandpass = [1.0, 10.0]
maxlag = 80
wiggles = False
global_normalization = False

figpath = None

# DONT EDIT BELOW THIS LINE
# =========================
paths = glob(os.path.join(data_path, "*"))
paths.sort()

for dpath in paths:
    fig, ax = plot_daily_correlations(data_path=dpath,
        data_format=data_format,
        bandpass=bandpass,
        maxlag=maxlag,
        global_normalization=global_normalization,
        gain=1.0,
        alpha=0.5,
        wiggles=wiggles,
        showfig=False,
    )

    if not fig:
        continue

    fig.set_size_inches(24,15)

    if figpath:
        basename = os.path.basename(dpath)
        basename += ".png"
        figname = os.path.join(figpath,basename)
        fig.savefig(figname)
    else:
        plt.show()

    plt.close()
