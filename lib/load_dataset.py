#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2021/8/25 17:50
@desc: load datasets
"""
import numpy as np
import torch
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def load_st_dataset(dataset, feature):
    #output： data [T N D]

    if dataset == 'powerLoad':
        df_raw = pd.read_csv('./data/powerLoad/NYPowerLoad.csv')
        target = 'MVH'
        tStr = 'date'
        '''
        border1s = [0, 12 * 30 * 24 , 12 * 30 * 24 + 4 * 30 * 24]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[set_type]
        border2 = border2s[set_type]
        '''
        if feature == 'S':
            df_data = df_raw[target].values
            df_dTime = df_raw[tStr]

    if dataset == 'ETDataset':
        df_raw = pd.read_csv('./data/ETDataset/ETTh1.csv')
        target = 'OT'
        tStr = 'date'
        if feature == 'S':
            df_data = df_raw[target].values
            df_dTime = df_raw[tStr]

    if dataset == 'traffic':
        df_raw = pd.read_csv('./data/traffic/PEMS04.csv')
        target = 'OT'
        tStr = 'date'
        if feature == 'S':
            df_data = df_raw[target].values
            df_dTime = df_raw[tStr]

    data = df_data
    dTime = df_dTime
    dTime.columns = ['date']
    print('prepare data has done!')
    return torch.tensor(data), pd.DataFrame(dTime)


def load_st_dataset_DF(dataset, feature):
    # output： data [T N D]

    if dataset == 'powerLoad':
            df_raw = pd.read_csv('./data/powerLoad/NYPowerLoad.csv')
            #target = 'MVH'
            '''
            border1s = [0, 12 * 30 * 24 , 12 * 30 * 24 + 4 * 30 * 24]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            border1 = border1s[set_type]
            border2 = border2s[set_type]
            '''
            if feature == 'S':
                df_data = df_raw

    if dataset == 'ETDataset':
            df_raw = pd.read_csv('./data/ETDataset/ETTh1.csv')
            #target = 'OT'
            if feature == 'S':
                df_data = df_raw

    if dataset == 'traffic':
            df_raw = pd.read_csv('./data/traffic/PEMS04.csv')
            #target = 'OT'
            if feature == 'S':
                df_data = df_raw

    print('prepare data has done!')
    columns = df_data.columns
    data = df_data[[columns[0],columns[-1]]]

    return data

if __name__ == '__main__':
    station_name = 'JSFD001'
    start_time = '20190131'
    data = load_st_dataset('powerload')
    print('prepare data has done!')