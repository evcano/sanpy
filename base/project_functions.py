import datetime
import itertools
import numpy as np
import os
import pickle
import warnings
from obspy import read_inventory, UTCDateTime
from obspy.geodetics import gps2dist_azimuth


def scan_stations(metadata_path, cmpts, ignore_net=None, ignore_sta=None):
    """
    Each metadata file must contain information about only one station
    """
    for cmp in cmpts:
        assert(cmp in ["N","E","Z"]),("Incorrect component")

    acceptable_channels = ["BHN","BHE","BHZ","HHN","HHE","HHZ"]

    metadata_files = os.listdir(metadata_path)
    stations = {}

    for file_ in metadata_files:
        inv = read_inventory(os.path.join(metadata_path, file_))
        inv_net, inv_sta, inv_fmt = file_.split(".")

        # skip unwanted inventories
        if ignore_net and inv_net in ignore_net:
            continue

        if ignore_sta and f"{inv_net}.{inv_sta}" in ignore_sta:
            continue

        # check that there is only one station in the inventory
        unique_networks = list(set(inv.get_contents()["networks"]))
        unique_stations = list(set(inv.get_contents()["stations"]))

        if len(unique_networks) > 1 or len(unique_stations) > 1:
            warnings.warn(f"Skipping {file_}: there is metadata about "
                          f"more than one station.")
            continue

        # check that the metadata filename matches the station it contains
        if f"{inv_net}.{inv_sta}" != unique_stations[0].split(" ")[0]:
            warnings.warn(f"Skipping {file_}: the file name does not match "
                          f"the name of the station in the metadata.")
            continue

        # at this point we are sure the inventory contains information only
        # about one station, however, sometimes the station configuration
        # changed during its deployement, resulting in multiple network,
        # station, and channel objects: loop through them
        lon = []
        lat = []
        elv = []
        cha = []

        for net_obj in inv:
            for sta_obj in net_obj:
                lon.append(sta_obj.longitude)
                lat.append(sta_obj.latitude)
                elv.append(sta_obj.elevation)
                for cha_obj in sta_obj:
                    if cha_obj.code in acceptable_channels:
                        if cha_obj.code[-1] in cmpts:
                            cha.append(cha_obj.code)

        # check that the station coordinates did not change over its deployment
        if len(set(lon)) > 1 or len(set(lat)) > 1 or len(set(elv)) > 1:
            warnings.warn(f"Skipping {file_}: contains different coordinates "
                          f"for the same station.")
            continue

        # check that there are channels
        if not cha:
            warnings.warn(f"Skipping {file_}: no channels found.")

        # eliminate repeated channels
        cha = list(set(cha))

        # check that there is information for all the requested components
        avail_cmpts = []
        for c in cha:
            avail_cmpts.append(c[-1])
        avail_cmpts = list(set(avail_cmpts))

        if avail_cmpts.sort() != cmpts.sort():
            warnings.warn(f"Skipping {file_}: not all the requested components "
                          f"are available")
            continue

        code = f"{inv_net}.{inv_sta}"
        stations[code] = {"net":inv_net,
                          "sta":inv_sta,
                          "lon":lon[0],
                          "lat":lat[0],
                          "elv":elv[0],
                          "cha":cha,
                         }
    return stations


def scan_pairs(stations):
    """
    We list all station pairs including repetitions such as AA, AB, BA
    """
    pairs = {}

    stations_codes = list(stations.keys())
    stations_codes.sort()
    pairs_obj = itertools.product(stations_codes, stations_codes)

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


def scan_waveforms(data_path, stations):
    """
    List relative path of all waveforms files inside data_path
    and their time span

    Data time span is determined from the waveforms files name

    data_path structure: data_path/network/station/waveforms

    """

    waveforms_paths = []
    stations_recording_start = []
    stations_recording_end = []

    stations_codes = list(stations.keys())
    stations_codes.sort()

    for sta_code in stations_codes:
        net, sta = sta_code.split(".")

        sta_waveforms_files = os.listdir(os.path.join(data_path,net,sta))
        sta_waveforms_files.sort()

        # keep only waveforms from desired channels
        sta_waveforms_files = [x for x in sta_waveforms_files
                               if x.split(".")[2] in stations[sta_code]["cha"]]

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
        # determine file-name prefix for the day
        day1 = day.replace("-", "")
        day2 = UTCDateTime(day) + 86400
        day2 = str(day2.date)
        day2 = day2.replace("-", "")
        day_prefix = f"{day1}T000000Z.{day2}T000000Z"

        day_waveforms = []

        for i, wpath in enumerate(waveforms_paths):
            fname = os.path.basename(wpath)
            if day_prefix in fname:
                day_waveforms.append(waveforms_paths[i])

        day_waveforms.sort()
        waveforms_paths_perday[day] = day_waveforms

    # verify that all waveforms are listed
    counter = 0
    for key in waveforms_paths_perday.keys():
        counter += len(waveforms_paths_perday[key])

    if counter == len(waveforms_paths):
        return waveforms_paths_perday
    else:
        print("Error: not all waveforms were listed\n")
        return


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
