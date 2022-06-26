import datetime
import itertools
import numpy as np
import os
import pandas as pd
import pickle
from obspy import read_inventory, UTCDateTime
from obspy.geodetics import gps2dist_azimuth


def list_waveforms_perday(waveforms_paths, data_span):
    waveforms_paths_perday = {}

    for day in data_span:
        # determine file-name prefix for the day
        day1 = day.replace("-", "")
        day2 = UTCDateTime(day) + 86400
        day2 = str(day2.date)
        day2 = day2.replace("-", "")
        day_prefix = "%sT000000Z.%sT000000Z" % (day1, day2)

        waveforms = []

        for i, tmp in enumerate(waveforms_paths):
            net, sta, fname = tmp.split('/')

            if day_prefix in fname:
                waveforms.append(waveforms_paths[i])

        waveforms_paths_perday[day] = waveforms

    # verify all waveforms are listed
    counter = 0

    for key in waveforms_paths_perday.keys():
        counter += len(waveforms_paths_perday[key])

    if counter == len(waveforms_paths):
        return waveforms_paths_perday
    else:
        print('Something went wrong')
        return


def scan_waveforms(par):
    """
    List relative path of all waveforms files inside data_path
    and their time span

    Data time span is determined from the waveforms files name

    data_path structure: data_path/network/station/waveforms

    """

    waveforms_paths = []
    start_dates = []
    end_dates = []

    networks = [x for x in os.listdir(par['data_path'])
                if os.path.isdir(os.path.join(par['data_path'], x))]

    for net in networks:
        if par['ignore_net'] and net in par['ignore_net']:
            continue

        net_path = os.path.join(par['data_path'], net)

        stations = [x for x in os.listdir(net_path)
                    if os.path.isdir(os.path.join(net_path, x))]

        for sta in stations:
            if par['ignore_sta'] and sta in par['ignore_sta']:
                continue

            sta_path = os.path.join(net_path, sta)

            # list relative path
            sta_waveforms_files = os.listdir(sta_path)
            sta_waveforms_files.sort()

            sta_waveforms_paths = ["{}/{}/{}".format(net, sta, x)
                                   for x in sta_waveforms_files]
            sta_waveforms_paths.sort()

            waveforms_paths.extend(sta_waveforms_paths)

            # determine time span
            first_file = sta_waveforms_files[0].split(".")
            last_file = sta_waveforms_files[-1].split(".")

            start = first_file[4].split("T")[0]
            end = last_file[5].split("T")[0]

            start_dates.append(start)
            end_dates.append(end)

    waveforms_paths.sort()

    start_dates.sort()
    end_dates.sort()

    start = start_dates[0]
    end = end_dates[-1]

    d1 = UTCDateTime(start).date
    d2 = UTCDateTime(end).date
    days = [str(d1 + datetime.timedelta(days=x))
            for x in range((d2-d1).days + 1)]
    data_span = np.array(days)

    return waveforms_paths, data_span


def scan_stations(par):
    """
    Reads station metadata from ObsPy compatible files and returns
    a PANDAS dataframe listing the stations and their information

    Thre must be one metadata file per station and each file must
    contain information from one channel (component) only.

    """

    code = []
    cols = {'net': [],
            'sta': [],
            'cmp': [],
            'loc': [],
            'lat': [],
            'lon': [],
            'elv': []}

    metadata_files = os.listdir(par['metadata_path'])

    for file_name in metadata_files:
        inv = read_inventory(os.path.join(par['metadata_path'], file_name))

        # list unique channels
        channels = inv.get_contents()['channels']
        channels = list(set(channels))

        if len(channels) > 1:
            print("Inventory contains more than one channel.")

        net, sta, loc, cmp = channels[0].split(".")

        if par['ignore_net'] and net in par['ignore_net']:
            continue

        if par['ignore_sta'] and sta in par['ignore_sta']:
            continue

        code.append("%s.%s" % (net, sta))
        cols['net'].append(net)
        cols['sta'].append(sta)
        cols['cmp'].append(cmp)
        cols['loc'].append(loc)
        cols['lat'].append(inv[0][0].latitude)
        cols['lon'].append(inv[0][0].longitude)
        cols['elv'].append(inv[0][0].elevation)

    stations = pd.DataFrame(cols)
    stations.index = code
    stations.sort_index(inplace=True)

    return stations


def scan_pairs(par, stations):
    """
    Defines station pairs given a PANDAS dataframe with the
    stations information

    Returns a PANDAS dataframe with station pairs information

    """

    code = []
    cols = {'dis': [],
            'az': [],
            'baz': [],
            'correlations': []}

    pairs = itertools.combinations(stations.index, 2)

    for pair in pairs:
        s1 = pair[0]
        s1lat = stations['lat'][s1]
        s1lon = stations['lon'][s1]

        s2 = pair[1]
        s2lat = stations['lat'][s2]
        s2lon = stations['lon'][s2]

        d, a, b = gps2dist_azimuth(lat1=s1lat,
                                   lon1=s1lon,
                                   lat2=s2lat,
                                   lon2=s2lon)
        d /= 1000.0  # convert from m to km

        net1, sta1 = s1.split('.')
        net2, sta2 = s2.split('.')
        c = "%s.%s_%s.%s" % (net1, sta1, net2, sta2)

        code.append(c)
        cols['dis'].append(d)
        cols['az'].append(a)
        cols['baz'].append(b)
        cols['correlations'].append(0)

    pairs = pd.DataFrame(cols)
    pairs.index = code

    return pairs


def save_project(project, filename):
    try:
        if os.path.exists(filename):
            os.remove(filename)

        with open(filename, 'wb') as _file:
            pickle.dump(project, _file, -1)

    except Exception:
        print('An exception occurred; project not saved')

    return


def load_project(filename):
    with open(filename, 'rb', -1) as _file:
        project = pickle.load(_file)

    return project
