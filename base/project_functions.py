import datetime
import itertools
import numpy as np
import os
import pickle
from glob import glob
from obspy import read_inventory, UTCDateTime
from obspy.geodetics import gps2dist_azimuth


def scan_metadata(metadata_path, chans, ignore_sta=None):
    """
    Scan metadata files to list all the available stations and their
    coordinates. Each file must contain information only of one station.
    The file must be named after the corresponding station.

    NOTE: The existence of the instrument response is not checked as
    this is done by <remove_response> Obspy command.
    """
    stations = {}

    acceptable_networks = []
    for key in chans.keys():
        acceptable_networks.append(key)

    metadata_files = glob(os.path.join(metadata_path,"*.xml"))
    metadata_files.sort()
    assert(metadata_files),("no metadata files available")

    for file_path in metadata_files:
        file_ = os.path.basename(file_path)
        inv = read_inventory(file_path)
        inv_net, inv_sta, inv_fmt = file_.split(".")

        # skip unwanted files
        if inv_net not in acceptable_networks:
            continue

        if ignore_sta and f"{inv_net}.{inv_sta}" in ignore_sta:
            continue

        # check that there is only one station in the file
        unique_networks = list(set(inv.get_contents()["networks"]))
        unique_stations = list(set(inv.get_contents()["stations"]))

        assert(len(unique_networks) == 1 and len(unique_stations) == 1),(
            f"{file_} contains info about more than one station")

        # check that the file is named after the corresponding station
        assert(f"{inv_net}.{inv_sta}" == unique_stations[0].split(" ")[0]),(
             f"{file_} is not named after the corresponding station")

        # at this point we are sure the file contains information
        # about one station, however, sometimes the station configuration
        # changed during its deployement, resulting in multiple network,
        # station, and channel objects: loop through them
        lon = []
        lat = []
        elv = []

        for net_obj in inv:
            for sta_obj in net_obj:
                lon.append(sta_obj.longitude)
                lat.append(sta_obj.latitude)
                elv.append(sta_obj.elevation)

        # check that the station coordinates did not change over its deployment
        lon = list(set(lon))
        lat = list(set(lat))
        elv = list(set(elv))

        assert(len(lon) == 1 and len(lat) == 1 and len(elv) == 1),(
            f"{file_} contains different coordinates for the same station")

        code = f"{inv_net}.{inv_sta}"
        stations[code] = {"net":inv_net,
                          "sta":inv_sta,
                          "lon":lon[0],
                          "lat":lat[0],
                          "elv":elv[0],
                         }

    return stations


def scan_pairs(stations):
    """
    We list unique station pairs allowing repetition of stations, i.e.,
    AA, BB, ...
    """
    pairs = {}

    stations_codes = list(stations.keys())
    stations_codes.sort()
    pairs_obj = itertools.combinations_with_replacement(stations_codes,2)

    for pair in pairs_obj:
        s1 = pair[0]
        s1lon = stations[s1]['lon']
        s1lat = stations[s1]['lat']

        s2 = pair[1]
        s2lon = stations[s2]['lon']
        s2lat = stations[s2]['lat']

        dis, az, baz = gps2dist_azimuth(lat1=s1lat,
                                        lon1=s1lon,
                                        lat2=s2lat,
                                        lon2=s2lon)

        dis /= 1000.0  # convert from m to km

        code = f"{s1}_{s2}"
        pairs[code] = {"dis":dis,
                       "az":az,
                       "baz":baz,
                       "ncorr":0
                      }
    return pairs


def scan_waveforms(data_path, stations, chans):
    """
    List relative path of all waveforms files inside data_path
    and their time span

    Data time span is determined from the waveforms files name

    data_path structure: data_path/network/station/waveforms

    """
    stations_codes = list(stations.keys())
    stations_codes.sort()

    waveforms_paths = []
    stations_recording_start = []
    stations_recording_end = []

    for sta_code in stations_codes:
        net, sta = sta_code.split(".")

        sta_waveforms_files = os.listdir(os.path.join(data_path,net,sta))
        sta_waveforms_files.sort()

        # keep only waveforms from desired channels
        sta_waveforms_files = [x for x in sta_waveforms_files
                               if x.split(".")[2] in chans[net]]

        # list waveforms paths
        sta_waveforms_paths = [os.path.join(net,sta,x)
                               for x in sta_waveforms_files]
        sta_waveforms_paths.sort()

        waveforms_paths.extend(sta_waveforms_paths)

        # get the first and last day where the station recorded data
        first_waveform = sta_waveforms_files[0]
        last_waveform = sta_waveforms_files[-1]

        start_day = first_waveform.split(".")[4].split("T")[0]
        end_day = last_waveform.split(".")[5].split("T")[0]

        stations_recording_start.append(start_day)
        stations_recording_end.append(end_day)

    waveforms_paths.sort()
    stations_recording_start.sort()
    stations_recording_end.sort()

    # list the data span in a list of days
    start = stations_recording_start[0]
    end = stations_recording_end[-1]

    d1 = UTCDateTime(start).date
    d2 = UTCDateTime(end).date

    days = [str(d1 + datetime.timedelta(days=x))
            for x in range((d2-d1).days + 1)]
    data_span = np.array(days)

    return waveforms_paths, data_span


def list_waveforms_perday(waveforms_paths, data_span):
    waveforms_paths_perday = {}

    for day in data_span:
        waveforms_paths_perday[day] = []

    for wpath in waveforms_paths:
        fname = os.path.basename(wpath)
        net, sta, cha, loc, sdate, edate, fmt = fname.split(".")
        sday, _ = sdate.split("T")
        sday = f"{sday[0:4]}-{sday[4:6]}-{sday[6:]}"
        waveforms_paths_perday[sday].append(wpath)

    for day in data_span:
        waveforms_paths_perday[day].sort()

    return waveforms_paths_perday


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
