#Anna Wojciechowska <anna.wojciechowska@gmail.com>
#Oslo November 2023

import wave
import h5py

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Dataset:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self._read_data()
        
    def _read_data(self):
        self.timedata = 0
        self.signal = 0
        return

    def get_time_data(self):
        return self.timedata
        
    def get_signal_data(self):
        return self.signal
        
    def get_framerate(self):
        return self.framerate
        
    def plot_data(npoints):
        return 0
 

class WaveDataset(Dataset):
    def _read_data(self):
        with wave.open(self.data_file_path, 'rb') as wf:
            frames = wf.readframes(-1)
            self.signal = np.frombuffer(frames, dtype=np.int16)
            self.framerate = wf.getframerate()
            self.timedata = np.linspace(0, len(self.signal) / self.framerate, num=len(self.signal))

    def plot_data(self, n_secs = 0.01):
        #n_secs = 0.01 # [s]
        n_secs_points = int(n_secs * self.framerate)
        fig, ax = plt.subplots(figsize=(25, 5))
        _ = ax.plot(self.timedata[:n_secs_points], self.signal[:n_secs_points])

class HDF5_Dataset(Dataset):
    def _read_data(self):
        dataset = h5py.File(self.data_file_path, 'r')
        self.timedata = (dataset['t'])[0,:]
        self.signal = (dataset['eta'])[0,:]
        
    def plot_data(self, n_points = 200):
        fig, ax = plt.subplots(figsize=(25, 5))
        _ = ax.plot(self.timedata[:n_points], self.signal[:n_points])

class DaupnerDataset(Dataset):
    def _read_data(self):
        self.dataset = pd.read_csv(self.data_file_path, names=['raw_height'], header=None)
        sea_level=self.dataset.raw_height.mean()
        # the oryignal data is flipped since it is recorded from above by the ultrasound proble
        self.dataset['corrected_height'] = sea_level - self.dataset.raw_height
        self._add_time_information()

    def _add_time_information(self):
        first_timestamp_str = "1995-01-01T15:00:00.000"
        time_delta_us = 468800
        time_delta = np.timedelta64(time_delta_us, 'us')
        timestamps_array = np.empty(self.dataset.shape[0], dtype='datetime64[us]')
        timestamps_array[0] = np.datetime64(first_timestamp_str)
        indices = np.arange(self.dataset.shape[0] - 1)
        timestamps_array[1:] = timestamps_array[0] + time_delta * (indices + 1)
        self.dataset['timestamp'] = timestamps_array

    def get_time_data(self):
        return self.dataset['timestamp']
        
    def get_signal_data(self):
        return self.dataset['corrected_height']
    
    #to do add npoints
    def plot_data(self, npoints=0):
        fig, ax = plt.subplots(figsize=(14, 10))
        date_format = mdates.DateFormatter('%H:%M:%s')
        id_max = self.dataset['corrected_height'].idxmax()
        id_min = self.dataset['corrected_height'].idxmin()
        max_sl = self.dataset.iloc[id_max].corrected_height
        min_sl = self.dataset.iloc[id_min].corrected_height
        timestamp_max = self.dataset.iloc[id_max].timestamp
        timestamp_min = self.dataset.iloc[id_min].timestamp
        plt.xticks(rotation=45)
        first = self.dataset['timestamp'].iloc[0]

        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=120))
        ax.xaxis.set_major_formatter(date_format)
        #plt.xticks([first, timestamp_min, timestamp_max], [f' {first}', f'min: {timestamp_min}', f'max: {timestamp_max}'])

        plt.ylim(min_sl - 1,max_sl + 1)
        plt.ylabel('surface level [m]')
        plt.yticks([min_sl, -5, 0, 5, 10, 15, max_sl], [f'min: {min_sl}',  '-5','0','5','10','15', f'max: {max_sl}'])

        ax.plot(self.dataset.timestamp, self.dataset.corrected_height, linewidth=1, label ='surface level')
        ax.axhline(y=0, color='red', linewidth=1, linestyle='-', label=f'sea level')

        ax.grid()
        ax.legend()

        plt.title('Daupner E data')
        plt.show()

class ExperimentDataset(Dataset):
    def _read_data(self):
        self.dataset = pd.read_csv(self.data_file_path, names = ['date_string', 'probe_1_raw', 'probe_2_raw', 'probe_3_raw', 'probe_4_raw', 'sensors'])
        date_format = '%m/%d/%Y %H:%M:%S.%f'
        self.dataset['date_time'] = pd.to_datetime(self.dataset['date_string'], format=date_format)
        self.dataset['elapsed_secs'] =  (self.dataset['date_time'] - self.dataset.at[0, 'date_time']).dt.total_seconds()
        #double check if the data should be flipped
        self.dataset['probe_4'] = self.dataset['probe_4_raw'].mean() - self.dataset['probe_4_raw']

    def get_time_data(self):
        return self.dataset['elapsed_secs']
        
    def get_signal_data(self):
        return self.dataset['probe_4']
    
   #to do add npoints
    def plot_data(self, npoints=0):
        fig, ax = plt.subplots(figsize=(20, 5))
        _ = ax.plot(self.dataset.elapsed_secs.values,self.dataset.probe_4)

 


  
