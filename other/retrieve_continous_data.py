# retrieve continous data using FDSN services

import numpy as np
import obspy
from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader

# define location coordinates
domain = RectangularDomain(minlatitude=17.8229, maxlatitude=20.8701,
                           minlongitude=-106.0945, maxlongitude=-101.4912)

# seismic networks
networks = ["XF","ZA"]

# seismic networks loop
for i in range(0,2):
    # read stations list
    stationsFile = networks[i]+"_stations.txt"
    stations = np.genfromtxt(stationsFile,dtype='str',comments='#')
    stations = stations.tolist()
    stations = ",".join(stations)

    print("Network: "+networks[i])
    print("Stations: "+stations)

    # define data restrictions
    restrictions = Restrictions(
        starttime=obspy.UTCDateTime(2006,1,1),
        endtime=obspy.UTCDateTime(2007,1,1),
        chunklength_in_sec=86400,
        network=networks[i],
        station=stations,
        location="30",
        channel="BHZ",
        reject_channels_with_gaps=False,
        minimum_length=0.0,
        sanitize=False,
        minimum_interstation_distance_in_m=100.0,
        location_priorities=["30"])

    # download data
    mdl = MassDownloader(providers=["IRIS"])
    mdl.download(domain, restrictions, download_chunk_size_in_mb=50, threads_per_client=1,
		mseed_storage=("raw_data/{network}/{station}/{network}.{station}.{channel}.{location}.{starttime}.{endtime}.mseed"),
		stationxml_storage=("raw_data/{network}/{station}.xml"))
