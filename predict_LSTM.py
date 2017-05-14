# -*- coding:utf-8 -*-
import load_data
import plot
import statistics as stats
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from cleansing_data import normalization
from cleansing_data import de_normalization

'''
使用有状态LSTM神经网络模型进行预测
'''


# 为LSTM神经网络模型设置训练数据
def set_training_data():
    # 读取数据集
    length_temperature, time_temperature, temperature = load_data.read_temperature_history(is_full=True)
    length_load, time_load, load = load_data.read_load_history(is_full=True)
    # 归一化
    time_temperature = time_temperature.reshape(len(time_temperature), 4)
    mean_time, std_time, time_temperature = normalization(time_temperature)
    temperature = temperature.reshape(len(temperature), 11)
    mean, std, temperature = normalization(temperature)
    load = load.reshape(len(load), 20)
    mean_load, std_load, load = normalization(load, if_log=True)

    time_temperature = time_temperature.reshape((len(time_temperature), 1, 4))
    temperature = temperature.reshape((len(temperature), 1, 11))
    load = load.reshape((len(load), 1, 20))

    # 使用以前的用电负荷数据作为预测的输入
    load_for_inputs = np.copy(load)
    for i in range(len(load_for_inputs)-1, -1, -1):
        index = i-24*7
        if index < 0:
            index = i
        load_for_inputs[i] = np.copy(load_for_inputs[index])
    inputs_full = np.concatenate((temperature, time_temperature, load_for_inputs), axis=2)
    inputs_full = inputs_full.reshape((len(inputs_full), 1, 35))

    load = load.reshape(len(load), 20)
    # inputs_full = inputs_full[0:39414]
    inputs = inputs_full[0:len(load) - 24*7]
    # outputs_full = load[0:39414]
    outputs_full = load
    outputs = load[0:len(load) - 24*7]

    print "Full inputs shape:", inputs_full.shape
    print "Full outputs shape:", outputs_full.shape
    print "Input shape:", inputs.shape
    print "Output shape:", outputs.shape
    return mean_load, std_load, inputs_full, inputs, outputs_full, outputs


def main():
    # 获取训练数据
    mean_load, std_load, inputs_full, inputs, outputs_full, outputs = set_training_data()
    # 设置参数
    tsteps = 1
    batch_size = 2
    epochs = 10
    hidden_size = 100
    LSTM_laysers_num = 3

    # 构造LSTM神经网络模型
    print "Creating Model..."
    model = Sequential()
    model.add(LSTM(hidden_size,
                   batch_input_shape=(batch_size, tsteps, inputs.shape[2]),
                   return_sequences=True,
                   stateful=True))
    for i in range(2, LSTM_laysers_num):
        model.add(LSTM(hidden_size,
                       batch_input_shape=(batch_size, tsteps, hidden_size),
                       return_sequences=True,
                       stateful=True))
    model.add(LSTM(hidden_size,
                   batch_input_shape=(batch_size, tsteps, hidden_size),
                   return_sequences=False,
                   stateful=True))
    model.add(Dense(outputs.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    print model.summary()

    # 训练模型
    print "Training..."
    for i in range(epochs):
        print 'Epoch', i+1, '/', epochs
        model.fit(inputs,
                  outputs,
                  batch_size=batch_size,
                  verbose=2,
                  epochs=1,
                  shuffle=False)
        model.reset_states()

    # 进行预测
    print "Predicting..."
    model.reset_states()
    predicted_output = model.predict(inputs_full, batch_size=batch_size)

    predicted_output = de_normalization(mean_load, std_load, predicted_output, if_log=True)
    expected_output = de_normalization(mean_load, std_load, outputs_full, if_log=True)

    # 计算20个区域总的用电负荷
    predicted_temp = np.zeros((len(predicted_output), 1))
    expected_temp = np.zeros((len(expected_output), 1))
    for i in range(len(predicted_output)):
        predicted_temp[i][0] = np.sum(predicted_output[i])
        expected_temp[i][0] = np.sum(expected_output[i])
    predicted_output = np.concatenate((predicted_output, predicted_temp), axis=1)
    expected_output = np.concatenate((expected_output, expected_temp), axis=1)

    # 将2008/06/17-2008/06/30用电负荷数据作为测试数据
    predict_future = predicted_output[-1-7*2*24-18:-18]
    expect_future = expected_output[-1-7*2*24-18:-18]
    predict_future = predict_future.reshape(len(predict_future), 21)
    expect_future = expect_future.reshape(len(expect_future), 21)

    # 计算误差
    for i in range(0, 21):
        mse = stats.mean_squared_error(list(expect_future[:, i]), list(predict_future[:, i]))
        nrmse = stats.normalized_rmse(list(expect_future[:, i]), list(predict_future[:, i]))
        mape = stats.mape(list(expect_future[:, i]), list(predict_future[:, i]))
        print "zone", i + 1, ":", "mse:", mse, "  nrmse:", nrmse, "  mape:", mape

    # 可视化制图
    plot.plot_comparison(expect_future, predict_future, plot_num=21, algorithm="(LSTM)")


if __name__ == "__main__":
    main()