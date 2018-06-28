# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import load_data

'''
对数据集进行清洗、格式化等预处理
'''


# 填充缺失的数据
def fill_missing_data():
    # 获取历史用电负荷数据
    length_load, time_load, load = load_data.read_load_history(is_full=True)
    # 获取历史温度数据
    length_temperature, time_temperature, temperature = load_data.read_temperature_history()

    print('Filling missing data...')
    time_temperature = time_temperature.reshape((len(time_temperature), 4))
    temperature = temperature.reshape((len(temperature), 11))
    load = load.reshape((len(load), 20))

    # 分别记录填充缺失值后的时间戳、温度和用电负荷数据
    time_temperature_full = time_temperature[0:39414]
    temperature_full = temperature[0:39414]
    load_full = load[0:39414]
    step_size = 365*24
    for i in range(168+18):
        index = len(temperature_full)
        time = np.zeros((1, 4))
        time = np.add(time, time_temperature_full[index-step_size-24])
        time[0][0] += 1
        time_temperature_full = np.append(time_temperature_full, time, axis=0)

        load_tmp = np.zeros((1, 20))
        load_tmp = np.add(load_tmp, load_full[index-7*24])
        load_full = np.append(load_full, load_tmp, axis=0)

        temperature_tmp = np.zeros((1, 11))
        for j in range(1, 5):
            prev_index = index - j*step_size - 24
            prev_temp = temperature_full[prev_index]
            temperature_tmp = np.add(temperature_tmp, prev_temp)
        temperature_tmp = temperature_tmp/4
        temperature_full = np.append(temperature_full, temperature_tmp, axis=0)
    # 写入csv文件
    df_time = pd.DataFrame(time_temperature_full)
    df_time.to_csv('./dataset/time_full.csv', header=False, index=False)
    df_temperature = pd.DataFrame(temperature_full)
    df_temperature.to_csv('./dataset/temperature_full.csv', header=False, index=False)
    df_load = pd.DataFrame(load_full)
    df_load.to_csv('./dataset/load_full.csv', header=False, index=False)
    print('Done')


# 数据归一化处理
def normalization(data, if_mean=True, if_std=True, if_log=False):
    # 取对数
    if if_log:
        data = np.log(data+1)
    df = pd.DataFrame(data=data)
    result = df
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    if if_mean:
        result = result.sub(mean, axis=1)
    if if_std:
        result = result.div(std, axis=1)
    return np.array(mean), np.array(std), np.array(result)


# 归一化数据的还原
def de_normalization(mean, std, data, if_mean=True, if_std=True, if_log=False):
    df = pd.DataFrame(data=data, copy=True)
    result = df
    if if_mean:
        result = result.mul(std, axis=1)
    if if_std:
        result = result.add(mean, axis=1)
    result = np.array(result)
    if if_log:
        scale_array = np.empty_like(result)
        scale_array.fill(np.e)
        result = np.power(scale_array, result)-1
    return result


if __name__ == "__main__":
    fill_missing_data()
