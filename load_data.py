# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os
'''
读取用电数据集和温度数据集
'''
LOAD_DATA_PATH = './dataset/load_history.csv'
TEMPERATURE_DATA_PATH = './dataset/temperature_history.csv'


# 读取历史用电负荷数据
def read_load_history(is_full=False):
    load_file = './dataset/load.csv'
    time_file = './dataset/time_load.csv'
    if is_full:
        load_file = './dataset/load_full.csv'
        time_file = './dataset/time_full.csv'
    if os.path.exists(load_file) and os.path.exists(time_file):
        load = np.array(pd.read_csv(load_file, header=None))
        time = np.array(pd.read_csv(time_file, header=None))
        if len(time) != len(load):
            raise ValueError("TypeError")
        length = len(load)
        time = time.reshape(length, 1, 4)
        load = load.reshape(length, 1, 20)
        return length, time, load
    else:
        return _read_raw_load(is_full)


# 读取历史温度数据
def read_temperature_history(is_full=False):
    temperature_file = './dataset/temperature.csv'
    time_file = './dataset/time_temperature.csv'
    if is_full:
        temperature_file = './dataset/temperature_full.csv'
        time_file = './dataset/time_full.csv'
    if os.path.exists(temperature_file) and os.path.exists(time_file):
        temperature = np.array(pd.read_csv(temperature_file, header=None))
        time = np.array(pd.read_csv(time_file, header=None))
        if len(time) != len(temperature):
            raise ValueError("TypeError")
        length = len(time)
        time = time.reshape((length, 1, 4))
        temperature = temperature.reshape((length, 1, 11))
        return length, time, temperature
    else:
        return _read_raw_temperature()


# 读取原始用电数据集并进行分割
def _read_raw_load(is_full=False):
    print('Loading electricity load data...')
    raw_load_path = LOAD_DATA_PATH
    if is_full:
        raw_load_path = './dataset/full_load_history.csv'
    raw_data = pd.read_csv(raw_load_path, sep=',')
    # 记录时间戳
    time = []
    # 记录用电负荷
    load = []
    for name, group in raw_data.groupby('zone_id'):
        if name == 1:
            for row in group.iterrows():
                timestamp = np.zeros((4,1),dtype=float)
                for i in range(1,4):
                    timestamp[i-1][0] = row[1][i]
                for i in range(4, len(row[1])):
                    load_temp = np.zeros((20,1), dtype=float)
                    if row[1][i] != np.nan:
                        timestamp_temp = np.copy(timestamp)
                        timestamp_temp[3][0] = i - 3
                        if isinstance(row[1][i], str):
                            load_temp[0][0] = float(row[1][i].replace(',', ''))
                        else:
                            load_temp[0][0] = float(row[1][i])
                        time.append(timestamp_temp)
                        load.append(load_temp)
        else:
            index = int(name)
            count = 0
            for row in group.iterrows():
                for i in range(4, len(row[1])):
                    if row[1][i] != np.nan:
                        if isinstance(row[1][i], str):
                            load[count][index - 1][0] = float(row[1][i].replace(',', ''))
                        else:
                            load[count][index - 1][0] = float(row[1][i])
                        count += 1
    if len(load) != len(time):
        raise ValueError("TypeError")
    length = len(time)
    time = np.array(time, dtype=np.str)
    load = np.array(load, dtype=np.float64)
    # 写入csv文件
    _load_to_csv(length, time, load, is_full)

    time = time.reshape(length, 1, 4)
    load = load.reshape(length, 1, 20)
    print('Done')
    return length, time, load


# 将用电数据集和时间序列分别写入到不同csv文件
def _load_to_csv(length, time, load, is_full=False):
    load_file = './dataset/load.csv'
    time_file = './dataset/time_load.csv'
    if is_full:
        load_file = './dataset/load_full.csv'
        time_file = './dataset/time_load_full.csv'
    time_df = pd.DataFrame(time.reshape(length, 4))
    load_df = pd.DataFrame(load.reshape(length, 20))
    # 写入csv文件
    time_df.to_csv(time_file, header=False, index=False)
    load_df.to_csv(load_file, header=False, index=False)


# 读取原始温度数据集并进行分割
def _read_raw_temperature():
    print('Loading temperature data...')
    raw_data = pd.read_csv(TEMPERATURE_DATA_PATH, sep=',')
    # 记录温度
    temperature = []
    # 记录时间戳
    time = []
    for name, group in raw_data.groupby('station_id'):
        if name == 1:
            for row in group.iterrows():
                timestamp = np.zeros((4,1), dtype=float)
                for i in range(1,4):
                    timestamp[i-1][0] = row[1][i]
                for i in range(4, len(row[1])):
                    temp = np.zeros((11,1), dtype=float)
                    if row[1][i] != np.nan:
                        temp[0][0] = row[1][i]
                        timestamp_temp = np.copy(timestamp)
                        timestamp_temp[3][0] = i-3
                        time.append(timestamp_temp)
                        temperature.append(temp)
        else:
            index = int(name)
            count = 0
            for row in group.iterrows():
                for i in range(4, len(row[1])):
                    if row[1][i] != np.nan:
                        temperature[count][index-1][0] = row[1][i]
                        count += 1
    if len(time) != len(temperature):
        raise ValueError("TypeError")
    length = len(time)
    time = np.array(time, dtype=np.str)
    temperature = np.array(temperature, dtype=np.float64)
    # 写入csv文件
    _temperature_to_csv(length, time, temperature)
    time = time.reshape(length, 1, 4)
    temperature = temperature.reshape(length, 1, 11)
    print('Done')
    return length, time, temperature


# 将温度数据集和时间序列分别写入到不同csv文件
def _temperature_to_csv(length, time, temperature):
    temperature_file = './dataset/temperature.csv'
    time_file = './dataset/time_temperature.csv'
    time_df = pd.DataFrame(time.reshape(length, 4))
    temperature_df = pd.DataFrame(temperature.reshape(length, 11))
    # 写入csv文件
    time_df.to_csv(time_file, header=False, index=False)
    temperature_df.to_csv(temperature_file, header=False, index=False)


if __name__ == "__main__":
    length_load, time_load, load_load = read_load_history()
    length_temp, time_temp, temp_temp = read_temperature_history()