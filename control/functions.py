import numpy as np
import os


def filter_station_pairs(P, thr_ndays, thr_snr, thr_cc, thr_nseasons):
    all_pairs = P.pairs.index
    filtered_pairs = []

    for pair in all_pairs:
        # check number of daily observations stacked
        log_file = os.path.join(P.par['data_path'], 'log', 'all',
                                '{}.log'.format(pair))

        ndays = np.genfromtxt(log_file)

        if ndays < thr_ndays:
            continue

        log_file = os.path.join(P.par['log_path'], '{}.log'.format(pair))

        seasonal_snr_cc = np.genfromtxt(log_file)
        nseasons = seasonal_snr_cc.shape[0]

        if P.par['data_type'] == 'correlation':
            # check snr
            pos_snr = seasonal_snr_cc[0, 0]
            neg_snr = seasonal_snr_cc[0, 2]

            if pos_snr < thr_snr and neg_snr < thr_snr:
                continue

            # check seasonal variation
            pos_counter = 0
            neg_counter = 0

            for i in range(1, nseasons):
                pos_cc = seasonal_snr_cc[i, 1]
                neg_cc = seasonal_snr_cc[i, 3]

                if pos_cc >= thr_cc:
                    pos_counter += 1

                if neg_cc >= thr_cc:
                    neg_counter += 1

            if pos_counter < thr_nseasons and neg_counter < thr_nseasons:
                continue

            filtered_pairs.append(pair)

        elif P.par['data_type'] == 'green':
            # check snr
            snr = seasonal_snr_cc[0, 0]

            if snr < thr_snr:
                continue

            # check seasonal variation
            counter = 0

            for i in range(1, nseasons):
                cc = seasonal_snr_cc[i, 1]

                if cc >= thr_cc:
                    counter += 1

            if counter < thr_nseasons:
                continue

            filtered_pairs.append(pair)

    return filtered_pairs


def reduce_virtual_sources(pairs_list):
    """
    Reorganize station pairs to reduce the number of virtual sources

    Loops over sources, set source as receiver and evaluates if the number
    of sources was reduced

    Certainly this sorting algorithm can be improved
    """

    sta1 = []
    sta2 = []

    for pair in pairs_list:
        a, b = pair.split('_')
        sta1.append(a)
        sta2.append(b)

    sta1 = np.array(sta1)
    sta2 = np.array(sta2)
    src = np.unique(sta1)

    nsrc0 = src.size

    # sources loop
    for s in src:
        tmp_sta1 = sta1.copy()
        tmp_sta2 = sta2.copy()

        rec_idx = np.where(tmp_sta1 == s)[0]

        # set source as receiver
        for idx in rec_idx:
            tmp_sta1[idx] = tmp_sta2[idx].copy()
            tmp_sta2[idx] = s.copy()

        # if the number of sources is reduced, keep changes
        if np.unique(tmp_sta1).size < np.unique(sta1).size:
            sta1 = tmp_sta1.copy()
            sta2 = tmp_sta2.copy()

    nsrc1 = np.unique(sta1).size
    print('Virtual sources before reorganization {}'.format(nsrc0))
    print('Virtual sources after reorganization {}'.format(nsrc1))

    pairs_list2 = []

    for idx in range(0, sta1.size):
        pairs_list2.append('{}_{}'.format(sta1[idx], sta2[idx]))

    pairs_list2.sort()

    return pairs_list2
