# -*- coding:utf-8 -*-
import datetime
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import load_data

'''
对数据进行可视化制图
'''


def plot_origin(length, time, data, plot_num, plot_name, y_axis):
    date_list = []
    date_steps = 24*7
    for x in range(length):
        year = int(time[x][0][0])
        month = int(time[x][0][1])
        day = int(time[x][0][2])
        date = datetime.date(year=year, month=month, day=day)
        date_list.append(date)
        x += date_steps
    font = FontProperties(fname='C:\Windows\Fonts\simsun.ttc', size=12)
    font_title = FontProperties(fname='C:\Windows\Fonts\simsun.ttc', size=14)
    for i in range(plot_num):
        index = i+1
        y_data = []
        for j in range(length):
            y_data.append(data[j][0][i])
            j += date_steps
        plt.plot_date(x=date_list, y=y_data, fmt='blue')
        plt.title(plot_name+str(index), fontproperties=font_title)
        plt.ylabel(y_axis, fontproperties=font)
        plt.xlabel(u"时间", fontproperties=font)
        plt.show()


def plot_comparison(expected_output, predicted_output, plot_num=21, algorithm="null"):
    if plot_num > len(predicted_output):
        plot_num = len(predicted_output)
    font = FontProperties(fname='C:\Windows\Fonts\simsun.ttc', size=12)
    font_title = FontProperties(fname='C:\Windows\Fonts\simsun.ttc', size=14)
    plot_name = u"2008/06/17-2008/06/30预测用电负荷 vs.实际用电负荷：zone-"
    legend_name = [u"实际用电负荷", u"预测用电负荷"+str(algorithm)]
    for i in range(plot_num):
        index = i + 1
        legend_var = []
        y_1, = plt.plot(list(expected_output[:, i]), color='blue')
        y_2, = plt.plot(list(predicted_output[:, i]), color='green')
        legend_var.append(y_1)
        legend_var.append(y_2)
        plt.title(plot_name + str(index), fontproperties=font_title)
        plt.legend(legend_var, legend_name, prop=font)
        plt.ylabel(u"用电量/kWh", fontproperties=font)
        plt.xlabel(u"时间/h", fontproperties=font)
        plt.show()


def plot_clustering(expected_output, predicted_output, algorithm="K-Means", zone_id=1):
    font = FontProperties(fname='C:\Windows\Fonts\simsun.ttc', size=12)
    font_title = FontProperties(fname='C:\Windows\Fonts\simsun.ttc', size=14)
    plot_name = u"2008/06/17-2008/06/30预测用电负荷 vs.实际用电负荷：zone-"
    legend_name = [u"实际用电负荷", u"预测用电负荷"+str(algorithm)]
    legend_var = []

    y_1, = plt.plot(expected_output, color='blue')
    y_2, = plt.plot(predicted_output, color='green')
    legend_var.append(y_1)
    legend_var.append(y_2)

    plt.title(plot_name + str(zone_id), fontproperties=font_title)
    plt.legend(legend_var, legend_name, prop=font)
    plt.ylabel(u"用电量/kWh", fontproperties=font)
    plt.xlabel(u"时间/h", fontproperties=font)
    plt.show()


def plot_feature(data_frame):
    plt.figure()
    data_frame.plot(kind='barh', x='feature', y='fscore')
    font = FontProperties(fname='C:\Windows\Fonts\simsun.ttc', size=12)
    font_title = FontProperties(fname='C:\Windows\Fonts\simsun.ttc', size=14)
    plt.title(u"XGBoost模型:特征重要性排名", fontproperties=font_title)
    plt.xlabel(u"相对重要性", fontproperties=font)
    plt.show()


if __name__ == "__main__":
    # 绘制原始数据曲线图
    length_load, time_load, load_load = load_data.read_load_history()
    plot_origin(length_load, time_load, load_load, 20, u"用电负荷历史数据：zone-", u"用电量/kWh")
    length_temp, time_temp, temp_temp = load_data.read_temperature_history()
    plot_origin(length_temp, time_temp, temp_temp, 11, u"历史温度数据：station-", u"温度/℃")