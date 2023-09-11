import itertools
import numpy as np
import os
from obspy import read
from scipy.fft import next_fast_len
from sanpy.base.project_functions import (list_waveforms_perday,
                                          scan_stations,
                                          scan_pairs,
                                          scan_waveforms)


class Preprocessing_Project(object):
    def __init__(self, par):
        if isinstance(par['ignore_net'], str):
            par['ignore_net'] = np.genfromtxt(par['ignore_net'],
                                              dtype=str).tolist()

        if isinstance(par['ignore_sta'], str):
            par['ignore_sta'] = np.genfromtxt(par['ignore_sta'],
                                              dtype=str).tolist()

        self.par = par

    def setup(self):
        self.stations = scan_stations(metadata_path=self.par["metadata_path"],
                                      cmpts=self.par["cmpts"],
                                      ignore_net=self.par["ignore_net"],
                                      ignore_sta=self.par["ignore_sta"]
                                     )

        self.waveforms_paths, self.data_span = scan_waveforms(
            data_path=self.par["data_path"],
            stations=self.stations
        )

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


class Correlation_Project(object):
    def __init__(self, par):
        if isinstance(par['ignore_net'], str):
            par['ignore_net'] = np.genfromtxt(par['ignore_net'],
                                              dtype=str).tolist()

        if isinstance(par['ignore_sta'], str):
            par['ignore_sta'] = np.genfromtxt(par['ignore_sta'],
                                              dtype=str).tolist()

        par['fs'] = 1. / par['dt']
        par['corr_overlap'] = par['corr_dur'] - par['corr_overlap']
        par['corr_npts'] = int((par['corr_dur'] * par['fs']) + 1)
        par['nfft'] = next_fast_len(2 * par['corr_npts'] - 1)
        par['out_npts'] = int((2. * par['maxlag'] * par['fs']) + 1)
        par['output_path'] = os.path.join(par['output_path'],
                                          'daily_correlations')
        par['psd_path'] = os.path.join(par['output_path'], 'psd')
        par['log_path'] = os.path.join(par['output_path'], 'log')

        self.par = par

    def setup(self):
        self.stations = scan_stations(metadata_path=self.par["metadata_path"],
                                      cmpts=self.par["data_cmpts"],
                                      ignore_net=self.par["ignore_net"],
                                      ignore_sta=self.par["ignore_sta"]
                                     )

        self.waveforms_paths, self.data_span = scan_waveforms(
            data_path=self.par["data_path"],
            stations=self.stations
        )

        self.waveforms_paths_perday = list_waveforms_perday(
            self.waveforms_paths, self.data_span)

        self.pairs = scan_pairs(self.stations)

        # at this point, all data is supposed to be processed and have the same
        # sampling rate, do a shallow check
        st = read(os.path.join(self.par['data_path'], self.waveforms_paths[0]))
        if st[0].stats['delta'] != self.par['dt']:
            print('Incorrect data sampling rate. Change it to {}.'.format(
                st[0].stats['delta']))

        # setup output directory
        for pair in self.pairs_list:
            tmp = os.path.join(self.par['output_path'], pair)
            if not os.path.isdir(tmp):
                os.makedirs(tmp)

        for station in self.stations_list:
            tmp = os.path.join(self.par['psd_path'], station)
            if not os.path.isdir(tmp):
                os.makedirs(tmp)

        if not os.path.isdir(self.par['log_path']):
            os.makedirs(self.par['log_path'])

    @property
    def stations_list(self):
        stalist = list(self.stations.keys())
        stalist.sort()
        return stalist

    @property
    def pairs_list(self):
        plist = list(self.pairs.keys())
        plist.sort()
        return plist

    @property
    def unique_pairs_list(self):
        plist = itertools.combinations_with_replacement(self.stations_list, 2)
        plist = [f"{p[0]}_{p[1]}" for p in plist]
        plist.sort()
        return plist

class Stacking_Project(object):
    def __init__(self, parameters):
        # set some parameters
        if isinstance(parameters['ignore_net'], str):
            parameters['ignore_net'] = np.genfromtxt(parameters['ignore_net'],
                                                     dtype=str).tolist()

        if isinstance(parameters['ignore_sta'], str):
            parameters['ignore_sta'] = np.genfromtxt(parameters['ignore_sta'],
                                                     dtype=str).tolist()

        parameters['corr_path'] = os.path.join(parameters['output_path'],
                                               'stacked_correlations')

        parameters['log_path'] = os.path.join(parameters['corr_path'], 'log')

        parameters['greens_path'] = os.path.join(parameters['output_path'],
                                                 'stacked_greens')

        self.par = parameters
        self.stations = scan_stations(self.par)
        self.pairs = scan_pairs(self.par, self.stations)

        # setup output directory
        if not os.path.isdir(self.par['corr_path']):
            os.makedirs(self.par['corr_path'])

        if not os.path.isdir(self.par['log_path']):
            os.makedirs(self.par['log_path'])

        if not os.path.isdir(self.par['greens_path']):
            os.makedirs(self.par['greens_path'])


class Control_Project(object):
    def __init__(self, parameters):
        # set some parameters
        if isinstance(parameters['ignore_net'], str):
            parameters['ignore_net'] = np.genfromtxt(parameters['ignore_net'],
                                                     dtype=str).tolist()

        if isinstance(parameters['ignore_sta'], str):
            parameters['ignore_sta'] = np.genfromtxt(parameters['ignore_sta'],
                                                     dtype=str).tolist()

        parameters['log_path'] = os.path.join(parameters['output_path'], 'log')

        tmp = os.listdir(parameters['data_path'])
        parameters['seasons'] = [d for d in tmp if d[0:2] == '20']

        self.par = parameters
        self.stations = scan_stations(self.par)
        self.pairs = scan_pairs(self.par, self.stations)

        # setup output directory
        if not os.path.isdir(self.par['log_path']):
            os.makedirs(self.par['log_path'])
