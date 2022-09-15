import os
from sanpy.base.functions import query_pairs, query_virtual_source
from sanpy.base.project_functions import load_project
from sanpy.util.plot import plot_correlations
from sanpy.util.plot import plot_greens


# PARAMETERS
# =========
project_path = '../projects/sub_cvc_stacking.pkl'
data_format = 'sac'
data_type = 'correlations'

season = 'all'
minfq = 1. / 10.
maxfq = 1. / 3.
virtual_source = None
conditions = ['0 < dis < 90']
apparent_velocity = True

maxlag = 80.0
global_normalization = False
yaxis = 'no'
amplitude_only = False

# DONT EDIT BELOW THIS LINE
# =========================
P = load_project(project_path)

# query observations
if conditions:
    pairs = query_pairs(P, conditions)
else:
    pairs = P.pairs.index.to_list()

if virtual_source:
    pairs = query_virtual_source(pairs, virtual_source)

# plot figures
if data_type == 'correlations':
    plot_correlations(data_path=os.path.join(P.par['corr_path'], season),
                      data_format=data_format,
                      pairs=pairs,
                      maxtime=maxlag,
                      bandpass=[minfq, maxfq],
                      global_normalization=global_normalization,
                      yaxis=yaxis,
                      amplitude_only=amplitude_only,
                      apparent_velocity=apparent_velocity)

elif data_type == 'green':
    plot_greens(data_path=os.path.join(P.par['greens_path'], season),
                data_format=data_format,
                pairs=pairs,
                maxtime=maxlag,
                bandpass=[minfq, maxfq],
                global_normalization=global_normalization,
                yaxis=yaxis,
                amplitude_only=amplitude_only,
                apparent_velocity=apparent_velocity)
