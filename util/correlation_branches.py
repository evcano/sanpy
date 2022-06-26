from obspy.io.sac.sactrace import SACTrace


def correlation_branches(tr, branch='all'):
    if branch == 'all':
        pos_flag = 1
        neg_flag = 1
        sym_flag = 1
    elif branch == 'pos':
        pos_flag = 1
        neg_flag = 0
        sym_flag = 0
    elif branch == 'neg':
        pos_flag = 0
        neg_flag = 1
        sym_flag = 0
    else:
        print('incorrect branch option')
        return

    dt = tr.stats.delta
    half_npts = (tr.stats.npts - 1) / 2
    maxlag = half_npts * dt

    if pos_flag == 1:
        tr_pos = tr.slice(starttime=(tr.stats.starttime + maxlag),
                          endtime=tr.stats.endtime)
        tr_pos = SACTrace.from_obspy_trace(tr_pos)
    if neg_flag == 1:
        tr_neg = tr.slice(starttime=tr.stats.starttime,
                          endtime=(tr.stats.starttime + maxlag))
        tr_neg = SACTrace.from_obspy_trace(tr_neg)

    # pos branch
    if pos_flag == 1:
        pos = tr_pos.copy()
        pos.b = 0
        pos.stla = tr_pos.stla  # station coor (receiver station)
        pos.stlo = tr_pos.stlo
        pos.stel = tr_pos.stel
        pos.evla = tr_pos.evla  # event coor (source station)
        pos.evlo = tr_pos.evlo
        pos.evdp = tr_pos.evdp
        pos.scale = 1
        pos.lcalda = 1
        pos.dist = tr_pos.dist
        pos.kevnm = tr_pos.kevnm  # event name (source station)
        pos.kstnm = tr_pos.kstnm  # station name (receiver station)

    # neg branch
    if neg_flag == 1:
        neg = tr_neg.copy()
        neg.data = neg.data[::-1]
        neg.b = 0
        neg.stla = tr_neg.evla
        neg.stlo = tr_neg.evlo
        neg.stel = tr_neg.evdp
        neg.evla = tr_neg.stla
        neg.evlo = tr_neg.stlo
        neg.evdp = tr_neg.stel
        neg.scale = 1
        neg.lcalda = 1
        neg.dist = tr_neg.dist
        neg.kevnm = tr_neg.kstnm  # event name (source station) 
        neg.kstnm = tr_neg.kevnm  # station name (receiver station)

    # symmetric
    if sym_flag == 1:
        sym = neg.copy()
        sym.data += pos.data
        sym.data /= 2.0

    if branch == 'pos':
        pos = pos.to_obspy_trace()
        return pos
    if branch == 'neg':
        neg = neg.to_obspy_trace()
        return neg
    if branch == 'all':
        pos = pos.to_obspy_trace()
        neg = neg.to_obspy_trace()
        sym = sym.to_obspy_trace()
        return pos, neg, sym
