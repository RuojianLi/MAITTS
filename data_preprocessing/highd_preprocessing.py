# Copyright (C) 2024 - 2024 Ruojian Li， Inc. All Rights Reserved 
# @Time    : 2024/3/8 21:30
# @Email   : ruojian_li@163.com
# License: MIT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from collections import Counter
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def get_length_id(df):
    temp_dic = {}
    for i in np.unique(df['id']):
        temp = df[df['id'] == i]['backSightDistance']
        temp_dic[i] = temp.max() - temp.min()
    return temp_dic


def get_space_time_df(all_data):
    data = all_data[['frame', 'backSightDistance', 'xVelocity']]
    data['backSightDistance'] = (data['backSightDistance'] / 4).round()
    data['frame'] = (data['frame'] / 50).round()
    temp_dic = {}
    for i in np.unique(data['frame']):
        temp = {}
        for index, row in data[data['frame'] == i].iterrows():
            try:
                if temp[row['backSightDistance']] > -1:
                    temp[row['backSightDistance']] = (temp[row['backSightDistance']] + row['xVelocity']) / 2
            except:
                temp[row['backSightDistance']] = row['xVelocity']
        temp_dic[i] = temp
    space_time_df = pd.DataFrame(temp_dic)
    space_time_df = space_time_df.sort_index().T
    space_time_df.columns = ["f_" + str(i * 4) for i in space_time_df.columns]
    space_time_df['Time'] = [i * 2 for i in space_time_df.index]
    space_time_df.reset_index(drop=True)
    return space_time_df


if __name__ == '__main__':
    dataset = pd.DataFrame()
    MISIING_RATE = 0.3
    for file_number in tqdm(range(1, 60)):  # 读取所有路段文件
        path = "highd-dataset-v1.0/data/" + str(f"{file_number:02d}") + "_tracks.csv"
        use_col = [
            'frame', 'id', 'x', 'y', 'xVelocity', 'yVelocity', 'xAcceleration',
            'yAcceleration', 'backSightDistance', 'laneId'
        ]
        df1 = pd.read_csv(path, usecols=use_col)

        for laneid in np.unique(df1['laneId']):  # 分车道存储信息
            temp_df = df1[df1['laneId'] == laneid]
            temp_df = temp_df[(temp_df['backSightDistance'] > 20)
                              & (temp_df['backSightDistance'] < 400)]
            temp_df = temp_df[(temp_df['frame'] > 100)
                              & (temp_df['frame'] < 22500)]

            if temp_df['xVelocity'].iloc[0] < 0:
                temp_df['xVelocity'] = -temp_df['xVelocity']
                temp_df['xAcceleration'] = -temp_df[
                    'xAcceleration']
            temp_id = str(file_number) + "_" + str(laneid)
            length_id = pd.Series(
                get_length_id(temp_df)).sort_values(ascending=False).index
            floating_car_id = length_id[:int(
                np.floor(len(np.unique(temp_df['id'])) * MISIING_RATE))]
            floating_car_df = temp_df[temp_df['id'].isin(floating_car_id)]
            other_car_df = temp_df[~temp_df['id'].isin(floating_car_id)]
            other_car_df = other_car_df[~(
                    (other_car_df['backSightDistance'] > 35) &
                    (other_car_df['backSightDistance'] < 385))]
            temp_df = pd.concat([floating_car_df, other_car_df])

            temp_dataset = get_space_time_df(temp_df)

            temp_dataset['RecordID'] = temp_id
            if len(temp_dataset.columns) == 98:
                dataset = pd.concat([dataset, temp_dataset])
            else:
                print("长度不匹配！")

    processed_data_set = pd.DataFrame()
    for i in np.unique(dataset['RecordID']):
        temp = dataset[dataset['RecordID'] == i]
        if len(temp) >= 400:
            processed_data_set = pd.concat([processed_data_set, temp.iloc[:400]])
    storage_path = "experimental_data/fixed_detector_floating_car_scene_data_" + str(MISIING_RATE)
    with open(storage_path, 'wb') as f:
        pickle.dump(processed_data_set, f)
