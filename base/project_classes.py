import itertools
import numpy as np
import os
from obspy import read
from scipy.fft import next_fast_len
from sanpy.base.project_functions import (list_waveforms_perday,
                                          scan_metadata,
                                          scan_pairs,
                                          scan_waveforms)

class Preprocessing_Project(object):
    def __init__(self, par):
        if par["ignore_sta"]:
            par["ignore_sta"] = np.genfromtxt(par["ignore_sta"],
                                              dtype=str).tolist()

        self.par = par

    def setup(self):
        self.stations = scan_metadata(metadata_path=self.par["metadata_path"],
                                      chans=self.par["data_chans"],
                                      ignore_sta=self.par["ignore_sta"]
                                     )

        self.waveforms_paths, self.data_span = scan_waveforms(
            data_path=self.par["data_path"],
            stations=self.stations,
            chans=self.par["data_chans"]
        )

        # setup output directory
        for code in self.stations_list:
            net, sta = code.split(".")
            dir_ = os.path.join(self.par['output_path'], net, sta)
            if not os.path.isdir(dir_):
                os.makedirs(dir_)

    @property
    def stations_list(self):
        stalist = list(self.stations.keys())
        stalist.sort()
        return stalist


class Correlation_Project(Preprocessing_Project):
    def __init__(self, par):
        if par["ignore_sta"]:
            par["ignore_sta"] = np.genfromtxt(par["ignore_sta"],
                                              dtype=str).tolist()

        par['fs'] = 1. / par['dt']
        par['corr_npts'] = int((par['corr_dur'] * par['fs']) + 1)
        par['corr_nfft'] = next_fast_len(2 * par['corr_npts'] - 1)
        par['save_npts'] = int((2. * par['maxlag'] * par['fs']) + 1)

        par['corr_path'] = os.path.join(par['output_path'], 'daily_corr')
        par['psd_path'] = os.path.join(par['output_path'], 'daily_psd')
        par['log_path'] = os.path.join(par['output_path'], 'log_correlation')

        par['data_cmpts'] = self._check_data_cmpts(par['data_chans'],par['corr_cmpts'])

        self.par = par

    def setup(self):
        self.stations = scan_metadata(metadata_path=self.par["metadata_path"],
                                      chans=self.par["data_chans"],
                                      ignore_sta=self.par["ignore_sta"]
                                     )

        self.waveforms_paths, self.data_span = scan_waveforms(
            data_path=self.par["data_path"],
            stations=self.stations,
            chans=self.par["data_chans"]
        )

        self.pairs = scan_pairs(self.stations)

        self.waveforms_paths_perday = list_waveforms_perday(
            self.waveforms_paths, self.data_span)

        # setup output directory
        for cmp in self.par["corr_cmpts"]:
            for pair in self.pairs_list:
                dir_ = os.path.join(self.par['corr_path'], cmp, pair)
                if not os.path.isdir(dir_):
                    os.makedirs(dir_)

        if self.par["save_psd"]:
            for cmp in self.par["data_cmpts"]:
                for station in self.stations_list:
                    dir_ = os.path.join(self.par['psd_path'], cmp, station)
                    if not os.path.isdir(dir_):
                        os.makedirs(dir_)

        if not os.path.isdir(self.par['log_path']):
            os.makedirs(self.par['log_path'])

    def _check_data_cmpts(self,data_chans, corr_cmpts):
        avail_options = ["EE","NN","ZZ","TT", "RR"]

        # based on the requested correlation components, check what data
        # components are required
        required_cmpts = []

        for cmp in corr_cmpts:
            assert(cmp in avail_options),(
                f"incorrect corr_cmpts; available options: {avail_options}")
            if cmp in ["EE", "NN", "ZZ"]:
                required_cmpts.append(cmp[0])
            elif cmp in ["RR", "TT"]:
                required_cmpts.extend(["E", "N"])

        required_cmpts = list(set(required_cmpts))

        # based on the read data channels, check that all required data
        # components are available
        for net in data_chans.keys():
            net_avail_cmpts = []
            for ch in data_chans[net]:
                net_avail_cmpts.extend(ch[-1])
            for cmp in required_cmpts:
                assert(cmp in net_avail_cmpts),(
                    f"data component {cmp} not available for network {net}")

        return required_cmpts

    @property
    def pairs_list(self):
        plist = list(self.pairs.keys())
        plist.sort()
        return plist


class Stacking_Project(Correlation_Project):
    def __init__(self, par):
        if par["ignore_sta"]:
            par["ignore_sta"] = np.genfromtxt(par["ignore_sta"],
                                              dtype=str).tolist()

        par['corr_path'] = os.path.join(par['output_path'], 'stacked_corr')
        par['greens_path'] = os.path.join(par['output_path'], 'stacked_greens')
        par['log_path'] = os.path.join(par['output_path'], 'log_stacking')

        self.par = par

    def setup(self):
        self.stations = scan_metadata(metadata_path=self.par["metadata_path"],
                                      chans=self.par["data_chans"],
                                      ignore_sta=self.par["ignore_sta"]
                                     )

        self.pairs = scan_pairs(self.stations)

        # setup output directory
        if not os.path.isdir(self.par["corr_path"]):
            os.makedirs(self.par["corr_path"])

        if self.par["compute_greens"]:
            if not os.path.isdir(self.par["greens_path"]):
                os.makedirs(self.par["greens_path"])

        if not os.path.isdir(self.par['log_path']):
            os.makedirs(self.par['log_path'])


class Control_Project(object):
    def __init__(self, par):
        par['log_path'] = os.path.join(par['output_path'], 'log_control')
        self.par = parameters

    def setup(self):
        self.stations = scan_metadata(metadata_path=self.par["metadata_path"],
                                      chans=self.par["data_chans"],
                                      ignore_sta=self.par["ignore_sta"]
                                     )

        self.pairs = scan_pairs(self.stations)

        # setup output directory
        if not os.path.isdir(self.par['log_path']):
            os.makedirs(self.par['log_path'])
