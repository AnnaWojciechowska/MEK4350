#Anna Wojciechowska <anna.wojciechowska@gmail.com>
#Oslo November 2023

import wave
import h5py

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#Dataset is an abstract class providing a uniform interface to handle all datasets in a consistent way
class Dataset:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self._read_data()
        self.frequency_resolution = self.sample_rate / self.number_of_samples()
        
    def _read_data(self):
        self.timedata = 0
        self.signal = 0


    @property 
    def _sample_rate(self):
        return self.sample_rate

    
    def number_of_samples(self):
        pass

    @property
    # sampling rate/the number of samples in the signal 
    def _frequency_resolution(self):
        return self.frequency_resolution
    
    def Nyqyist_frequency(self):
        return 0.5 * self.frequency_resolution
    
    def SWH(self, std = True):
        if std:
            signal = self.get_signal_data()
            if signal.dtype == np.int16:
                # in case of wave data I tranform it to int64 since there is overflow error when squaring 
                signal = (signal).astype(np.int64)
            N = len(signal)
            std = np.sqrt(sum(np.square(signal))/N)
            return 4*std
        
    def mean(self):
        return (self.get_signal_data()).mean()
    
    def get_time_data(self):
        return self.timedata
        
    def get_signal_data(self):
        return self.signal
      
        
    def plot_data(npoints):
        pass
    
# dataset I
class ExperimentDataset(Dataset):
    def _read_data(self):
        column_names = ['date_string', 'probe_1_raw', 'probe_2_raw', 'probe_3_raw', 'probe_4_raw', 'sensors']
        self.dataset = pd.read_csv(self.data_file_path, names = column_names)
        date_format = '%m/%d/%Y %H:%M:%S.%f'
        self.dataset['date_time'] = pd.to_datetime(self.dataset['date_string'], format=date_format)
        self.dataset['elapsed_secs'] =  (self.dataset['date_time'] - self.dataset.at[0, 'date_time']).dt.total_seconds()
        #double check if the data should be flipped
        #tutaj read description from Karsten oblig
        self.dataset['probe_4'] = self.dataset['probe_4_raw'].mean() - self.dataset['probe_4_raw']
        self.sample_rate = 125

    def get_time_data(self):
        return self.dataset['elapsed_secs']
        
    def get_signal_data(self):
        return self.dataset['probe_4']
     
    def number_of_samples(self):
        return self.dataset.shape[0]
    
   #to do add npoints
    def plot_data(self, npoints=0):
        fig, ax = plt.subplots(figsize=(20, 5))
        _ = ax.plot(self.dataset.elapsed_secs.values,self.dataset.probe_4, linewidth=1, label ='eta/free surface level')
        ax.axhline(y=0, color='red', linewidth=1, linestyle='-', label=f'water level')
        half_swh  =round(self.SWH()/2,2)
        ax.axhline(y=half_swh, color='navy', linewidth=1, linestyle='-', label=f'Hs')
        plt.yticks([-0.02, -0.01, 0, 0.01, half_swh, 0.02], ['-0.02', '-0.01','0', '0.01', f'Hs: {half_swh}',  '0.02',])
        plt.ylabel('surface level [m]')
        ax.grid()
        ax.legend()
        plt.title('Oslo wave lab experiment data')


# dataset II
class HDF5_Dataset(Dataset):
    def _read_data(self):
        dataset = h5py.File(self.data_file_path, 'r')
        self.timedata = (dataset['t'])[0,:]
        self.signal = (dataset['eta'])[0,:]
        #tutaj I dont know sample_rate
        self.sample_rate = 0 

    def number_of_samples(self):
        return len(self.signal)

        
    def plot_data(self, n_points = 200):
        fig, ax = plt.subplots(figsize=(25, 5))
        _ = ax.plot(self.timedata[:n_points], self.signal[:n_points],linewidth=1, label ='eta/free surface level')
        ax.axhline(y=0, color='red', linewidth=1, linestyle='-', label=f'sea level')
        half_swh  =round(self.SWH()/2,1)
        ax.axhline(y=half_swh, color='navy', linewidth=1, linestyle='-', label=f'Hs')
        plt.yticks([-1.5, -1, -0.5, 0, 0.5, 1, half_swh, 1.5 ], ['-1.5', '-1','0.5', '0', '0.5', '1', f'Hs: {half_swh}', '1.5'])
        plt.ylabel('surface level [m]')
        ax.grid()
        ax.legend()
        plt.title('Bay of Biscay data')


# dataset III
class DaupnerDataset(Dataset):
    def _read_data(self):
        self.dataset = pd.read_csv(self.data_file_path, names=['raw_height'], header=None)
        sea_level=self.dataset.raw_height.mean()
        # the oryignal data is flipped since it is recorded from above by the ultrasound proble
        self.dataset['corrected_height'] = sea_level - self.dataset.raw_height
        self._add_time_information()
        # number of samples / number of seconds (20 min * 60 seconds)
        self.sample_rate = self.dataset.shape[0]/(20*60)

    def number_of_samples(self):
        return self.dataset.shape[0]


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
    
    def get_data(self):
        return self.dataset
    
    #to do add npoints
    def plot_data(self, npoints=0):
        fig, ax = plt.subplots(figsize=(14, 10))
        date_format = mdates.DateFormatter('%H:%M:%s')
        id_max = self.dataset['corrected_height'].idxmax()
        id_min = self.dataset['corrected_height'].idxmin()
        max_sl = round(self.dataset.iloc[id_max].corrected_height,1)
        min_sl = round(self.dataset.iloc[id_min].corrected_height,1)
        #timestamp_max = self.dataset.iloc[id_max].timestamp
        #timestamp_min = self.dataset.iloc[id_min].timestamp
        plt.xticks(rotation=45)
        #first = self.dataset['timestamp'].iloc[0]

        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=120))
        ax.xaxis.set_major_formatter(date_format)
        #plt.xticks([first, timestamp_min, timestamp_max], [f' {first}', f'min: {timestamp_min}', f'max: {timestamp_max}'])
        half_swh = round(self.SWH()/2,1)
        ax.axhline(y=half_swh, color='navy', linewidth=1, linestyle='-', label=f'Hs')

        plt.ylim(min_sl - 1,max_sl + 1)
        plt.ylabel('surface level [m]')
        plt.yticks([min_sl, -5, 0, 5, half_swh, 10,  15, max_sl], [f'min: {min_sl}',  '-5','0','5',f'Hs: {half_swh}', '10','15', f'max: {max_sl}'])

        ax.plot(self.dataset.timestamp, self.dataset.corrected_height, linewidth=1, label ='sea level')
        ax.axhline(y=0, color='red', linewidth=1, linestyle='-', label=f'Hs')
        
       
        ax.grid()
        ax.legend()
        plt.title('Daupner E data')
        
 
# dataset IV
class WaveDataset(Dataset):
    def _read_data(self):
        with wave.open(self.data_file_path, 'rb') as wf:
            frames = wf.readframes(-1)
            self.signal = np.frombuffer(frames, dtype=np.int16)
            self.sample_rate = wf.getframerate()
            self.timedata = np.linspace(0, len(self.signal) / self.sample_rate, num=len(self.signal))
    
    def number_of_samples(self):
        return len(self.signal)

    def plot_data(self, n_secs = 0.01):
        #n_secs = 0.01 # [s]
        n_secs_points = int(n_secs * self.sample_rate)
        fig, ax = plt.subplots(figsize=(25, 5))
        _ = ax.plot(self.timedata[:n_secs_points], self.signal[:n_secs_points],  label ='sound level')
        ax.axhline(y=0, color='red', linewidth=1, linestyle='-', label=f'0 level')
        half_swh = round(self.SWH()/2,1)
        ax.axhline(y=half_swh, color='navy', linewidth=1, linestyle='-', label=f'swh')
        #plt.yticks( [-30000, -20000, -10000, 0, 10000, 20000, 30000], ['-30K','-20K','-10K', '0','10K','20K',,'30K', f'SWH: {half_swh}'])
        plt.yticks( [-30000, -20000, -10000, 0, 10000, 20000, 30000, half_swh], ['-30K','-20K','-10K', '0','10K','20K','30K', f'Hs: {half_swh}'])
        plt.ylabel('raw amplitude')
        ax.grid()
        ax.legend()
        plt.title('Sound file')
