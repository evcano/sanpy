import numpy as np
from obspy.signal.filter import envelope


def max_envelope_window(tr, minperiod, maxperiod, src_dist=None, vmin=None):

    # dominant period
    T = (minperiod + maxperiod) / 2.

    # window duration is 5 times the dominant period
    window_dur = 5.0 * T
    window_nsamp = window_dur // tr.stats.delta

    env = envelope(tr.data)

    # mute envelope after maximum expected arrival time plus one window
    if src_dist and vmin:
        maxtime = src_dist / vmin + window_dur
        maxtime = int(maxtime / tr.stats.delta)

        if maxtime < env.size:
            env[maxtime:] = 0.0

    # center window at maximum of envelope
    idx = np.argmax(env)
    window = [idx - int(window_nsamp / 2), idx + int(window_nsamp / 2)]

    window[0] = max(0, window[0])
    window[1] = min(tr.stats.npts-2, window[1])

    return window
