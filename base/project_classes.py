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
        return list(self.stations.keys())


class Correlation_Project(object):
    def __init__(self, parameters):
        # set some parameters
        if isinstance(parameters['ignore_net'], str):
            parameters['ignore_net'] = np.genfromtxt(parameters['ignore_net'],
                                                     dtype=str).tolist()

        if isinstance(parameters['ignore_sta'], str):
            parameters['ignore_sta'] = np.genfromtxt(parameters['ignore_sta'],
                                                     dtype=str).tolist()

        parameters['fs'] = 1. / parameters['dt']

        parameters['corr_overlap'] = (parameters['corr_dur'] -
                                      parameters['corr_overlap'])

        parameters['corr_npts'] = int((parameters['corr_dur'] *
                                       parameters['fs']) + 1)

        parameters['nfft'] = next_fast_len(2 * parameters['corr_npts'] - 1)

        parameters['out_npts'] = int((2. * parameters['maxlag'] *
                                      parameters['fs']) + 1)

        parameters['output_path'] = os.path.join(parameters['output_path'],
                                                 'daily_correlations')

        parameters['psd_path'] = os.path.join(parameters['output_path'], 'psd')
        parameters['log_path'] = os.path.join(parameters['output_path'], 'log')

        self.par = parameters
        self.stations = scan_stations(self.par)
        self.pairs = scan_pairs(self.par, self.stations)

        self.waveforms_paths, self.data_span = scan_waveforms(self.par)

        self.waveforms_paths_perday = list_waveforms_perday(
            self.waveforms_paths, self.data_span)

        # check data sampling rate
        st = read(os.path.join(self.par['data_path'], self.waveforms_paths[0]))

        if st[0].stats['delta'] != self.par['dt']:
            print('Incorrect data sampling rate. Change it to {}.'.format(
                st[0].stats['delta']))

        # setup output directory
        for pair in self.pairs.index:
            tmp = os.path.join(self.par['output_path'], pair)
            if not os.path.isdir(tmp):
                os.makedirs(tmp)

        for station in self.stations.index:
            tmp = os.path.join(self.par['psd_path'], station)
            if not os.path.isdir(tmp):
                os.makedirs(tmp)

        if not os.path.isdir(self.par['log_path']):
            os.makedirs(self.par['log_path'])


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
