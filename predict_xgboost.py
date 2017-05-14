# -*- coding:utf-8 -*-
import plot
import statistics as stats
import numpy as np
import pandas as pd
import xgboost as xgb


'''
使用XGBoost模型进行预测
'''


# 对用电负荷数据进行预处理
def pre_processing_data():
    print "Processing data..."
    load_file = './dataset/load_full.csv'
    time_file = './dataset/time_load.csv'
    temperature_file = './dataset/temperature_full.csv'

    # 读取用电负荷数据
    data_load = pd.read_csv(load_file, header=None)
    data_load = np.log(data_load + 1)
    column_name = 'zone'
    columns = []
    for i in range(1, 21):
        columns.append(column_name+str(i))
    data_load.columns = columns

    # 读取用电负荷对应的时间数据并加上标签
    time = pd.read_csv(time_file, header=None, dtype=int)
    time.columns = ["year", "month", "day", "hour"]
    for i in range(0, len(time)):
        if time.ix[i, 0] == 0:
            time.ix[i, 0] = '2004'
        elif time.ix[i, 0] == 1:
            time.ix[i, 0] = '2005'
        elif time.ix[i, 0] == 2:
            time.ix[i, 0] = '2006'
        elif time.ix[i, 0] == 3:
            time.ix[i, 0] = '2007'
        elif time.ix[i, 0] == 4:
            time.ix[i, 0] = '2008'
    time_load = time
    time_load['date'] = time_load.apply(
        lambda x: pd.datetime.strptime("{0} {1} {2}".format(x['year'], x['month'], x['day']), '%Y %m %d'),
        axis=1)
    time_load['date_time'] = pd.to_datetime(
        time_load['date'] + pd.TimedeltaIndex(time_load['hour'], unit='H'))

    # 读取温度数据并加上标签
    temperature = pd.read_csv(temperature_file, header=None)
    column_name = 'station'
    columns = []
    for i in range(1, 12):
        columns.append(column_name+str(i))
    temperature.columns = columns

    # 将时间-用电负荷-温度数据进行连接
    time_load_temperature = pd.concat([time_load, data_load, temperature], axis=1)

    # 将2008/06/24-2008/06/30用电负荷数据作为测试数据
    data_test = data_load.tail(354).head(2*7*24)
    data_test = np.exp(data_test) - 1

    print "Done"
    return time_load_temperature, data_test


def data_zone(time_load_temperature, i):
    data_slice = time_load_temperature.ix[:, [0, 1, 2, 3, 4, 5, 5 + i, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]]
    columns = ['year', 'month', 'day', 'hour','date', 'date_time', 'load']
    for j in range(1, 12):
        columns.append('station' + str(j))
    t = pd.DataFrame(data_slice)
    t.columns = columns
    t.loc[:, 'zone_id'] = [i] * len(t)
    load = pd.Series(t['load'])
    # 用电负荷时间滞后项
    lag_load = pd.concat([load.shift(j) for j in range(1, 25)], axis=1)
    lag_load.columns = ['lag_t-%d' % j for j in range(1, 25)]
    lag_load = pd.concat([t, lag_load], axis=1)
    lag_load = lag_load.tail(len(lag_load) - 24)
    return lag_load


# 为XGBoost模型设置训练数据
def set_training_data(time_load_temperature):
    print "Setting training data..."
    # 构建用电负荷数据时间滞后项特征
    combine_zones = data_zone(time_load_temperature, 1)
    for i in range(2, 21):
        zone_temp = data_zone(time_load_temperature, i)
        combine_zones = pd.concat([combine_zones, zone_temp], axis=0)

    data = combine_zones[combine_zones['zone_id'] == 1]
    data = data.head(39222)
    for i in range(2, 21):
        data_temp = combine_zones[combine_zones['zone_id'] == i]
        data_temp = data_temp.head(39222)
        data = pd.concat([data, data_temp], axis=0)

    # 获取特征和训练样本
    features = list(data.columns[19:])
    y_train = data.load
    x_train = data[features]
    print "Done"
    return features, x_train, y_train, combine_zones


# 构建特征图
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feature in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feature))
        i = i + 1
    outfile.close()


# 合并特征
def merge_features(combine_zones):
    # 转换分类变量
    data = combine_zones.ix[:,
           [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, -1]].reset_index()
    data = data.drop('index', axis=1)
    predict_by = 1
    predict_by = -(predict_by - 1)

    # 合并year特征
    years = pd.get_dummies(pd.DatetimeIndex(data.date_time).year, prefix='Year')
    years = years.set_index(pd.DatetimeIndex(data.date_time))
    if predict_by != 0:
        if predict_by > 0 or type(predict_by) != int:
            raise ValueError("ValueError")
        else:
            years = years.shift(predict_by)
    data = data.set_index('date_time')
    data = pd.concat([data, years], axis=1)

    # 合并month特征
    months = pd.get_dummies(pd.DatetimeIndex(data.index).month, prefix='Month')
    months = months.set_index(pd.DatetimeIndex(data.index))
    if predict_by != 0:
        if predict_by > 0 or type(predict_by) != int:
            raise ValueError("ValueError")
        else:
            months = months.shift(predict_by)
    data = pd.concat([data, months], axis=1)

    # 合并day特征
    days = pd.get_dummies(pd.DatetimeIndex(data.index).day, prefix='Day')
    days = days.set_index(pd.DatetimeIndex(data.index))
    if predict_by != 0:
        if predict_by > 0 or type(predict_by) != int:
            raise ValueError("ValueError")
        else:
            days = days.shift(predict_by)
    data = pd.concat([data, days], axis=1)

    # 合并hour特征
    hours = pd.get_dummies(pd.DatetimeIndex(data.index).hour, prefix='Hour')
    hours = hours.set_index(pd.DatetimeIndex(data.index))
    if predict_by != 0:
        if predict_by > 0 or type(predict_by) != int:
            raise ValueError("ValueError")
        else:
            hours = hours.shift(predict_by)
    data = pd.concat([data, hours], axis=1)
    return data


# 用测试数据进行预测
def prediction(index, x_test, x_test_lag, model):
    d_test = xgb.DMatrix(x_test)
    y_predict = model.predict(d_test)
    y_predict = y_predict[index]
    x_test_lag.ix[168*2+index, 4] = y_predict

    load = pd.Series(x_test_lag['load'])
    lag_load = pd.concat([load.shift(j) for j in [1,2,3,24]], axis=1)
    lag_load.columns = ['lag_t-%d' %j for j in [1,2,3,24]]
    lag_load = pd.concat([load, lag_load], axis=1)
    lag_load = lag_load.tail(len(lag_load)-168*2)

    test = x_test
    test['lag_t-1'] = lag_load['lag_t-1']
    test['lag_t-2'] = lag_load['lag_t-2']
    test['lag_t-3'] = lag_load['lag_t-3']
    test['lag_t-24'] = lag_load['lag_t-24']
    return y_predict, test, x_test_lag


# 判断不同天气站温度数据对不同区域用电负荷数据的影响
def matching_zone_station(data, zone_id, station_id):
    train = data[data['zone_id'] == zone_id]
    features = list(data.columns[17:])
    features.append(data.columns[station_id+4])

    x_train = train[features].head(39222)
    y_train = train.load.head(39222)
    x_test = train[features].tail(354+168*2).head((168+168)*2)
    x_test = x_test.tail(len(x_test) - 168*2)
    x_test_lag = data.tail(354+168*2).head((168+168)*2)

    # 设置XGBoost模型参数
    xgb_params = {"objective": "reg:linear",
                  "eta": 0.01,
                  "max_depth": 8,
                  "seed": 42,
                  "silent": 1}
    num_rounds = 1000
    d_train = xgb.DMatrix(x_train, label=y_train)
    model = xgb.train(xgb_params, d_train, num_rounds)

    # 对2008/06/17-2008/06/30的用电负荷进行预测
    predict = []
    for i in range(0, 168*2):
        y_predict, test, test_lag = prediction(i, x_test, x_test_lag, model)
        y_predict = np.exp(y_predict) - 1
        predict.append(y_predict)
        x_test = test
        x_test_lag = test_lag
    return predict


# 获取每个区域最匹配的天气站
def fit_zone_station(data, data_test, zone_id):
    mse = []
    for i in range(1, 12):
        predict = matching_zone_station(data, zone_id, i)
        # 计算Normalized Root Mean Square Error(NRMSE)
        # nrmse_temp = stats.normalized_rmse(data_test[:, zone_id-1], predict)
        mse_temp = ((predict-data_test.ix[:, zone_id-1])**2).mean()
        mse.append(mse_temp)
    print "zone", zone_id, ":"
    print mse
    fit_station = mse.index(min(mse)) + 1
    print "fit station:", fit_station

    return fit_station


def main():
    # 获取训练数据
    time_load_temperature, expect_future = pre_processing_data()
    features, x_train, y_train, combine_zones = set_training_data(time_load_temperature)

    # 构建特征图
    create_feature_map(features)
    # 设置XGBoost模型参数
    xgb_params = {"objective": "reg:linear",
                  "eta": 0.01,
                  "max_depth": 8,
                  "seed": 42,
                  "silent": 1}
    num_rounds = 500
    train_data = xgb.DMatrix(x_train, label=y_train)
    gbdt = xgb.train(xgb_params, train_data, num_rounds)

    # 对特征重要性进行排名
    importance = gbdt.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=op.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    # 显示特征相对排名
    plot.plot_feature(df)

    # 合并特征数据
    fit_data = merge_features(combine_zones)

    '''
    # Test
    pre1 = pd.DataFrame(matching_zone_station(fit_data, 1, 2))
    pre2 = pd.DataFrame(matching_zone_station(fit_data, 2, 11))
    pre3 = pd.DataFrame(matching_zone_station(fit_data, 3, 11))
    pre4 = pd.DataFrame(matching_zone_station(fit_data, 4, 9))
    pre5 = pd.DataFrame(matching_zone_station(fit_data, 5, 11))
    pre6 = pd.DataFrame(matching_zone_station(fit_data, 6, 11))
    pre7 = pd.DataFrame(matching_zone_station(fit_data, 7, 11))
    pre8 = pd.DataFrame(matching_zone_station(fit_data, 8, 2))
    pre9 = pd.DataFrame(matching_zone_station(fit_data, 9, 6))
    pre10 = pd.DataFrame(matching_zone_station(fit_data, 10, 1))
    pre11 = pd.DataFrame(matching_zone_station(fit_data, 11, 1))
    pre12 = pd.DataFrame(matching_zone_station(fit_data, 12, 3))
    pre13 = pd.DataFrame(matching_zone_station(fit_data, 13, 2))
    pre14 = pd.DataFrame(matching_zone_station(fit_data, 14, 4))
    pre15 = pd.DataFrame(matching_zone_station(fit_data, 15, 8))
    pre16 = pd.DataFrame(matching_zone_station(fit_data, 16, 3))
    pre17 = pd.DataFrame(matching_zone_station(fit_data, 17, 6))
    pre18 = pd.DataFrame(matching_zone_station(fit_data, 18, 3))
    pre19 = pd.DataFrame(matching_zone_station(fit_data, 19, 6))
    pre20 = pd.DataFrame(matching_zone_station(fit_data, 20, 11))
    predict_future = pd.concat([pre1,pre2,pre3,pre4,pre5,pre6,pre7,pre8,pre9,pre10,
                      pre11,pre12,pre13,pre14,pre15,pre16,pre17,pre18,pre19,pre20], axis=1)
    '''

    predict_future = pd.DataFrame()
    for zone_id in range(1, 21):
        fit_station = fit_zone_station(fit_data, expect_future, zone_id)
        predict = matching_zone_station(fit_data, zone_id, fit_station)
        predict = pd.DataFrame(predict)
        predict_future = pd.concat([predict_future, predict], axis=1)

    predict_future = predict_future.ix[:, [0, 11, 13, 14, 15, 16, 17, 18, 19, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]]
    predict_future = pd.DataFrame(predict_future)

    # 计算20个区域总的用电负荷
    predict_total = np.zeros((len(predict_future), 1))
    expect_total = np.zeros((len(predict_future), 1))
    for i in range(len(predict_future)):
        predict_total[i][0] = np.sum(predict_future.iloc[i])
        expect_total[i][0] = np.sum(expect_future.iloc[i])
    predict_future = np.concatenate((predict_future, predict_total), axis=1)
    expect_future = np.concatenate((expect_future, expect_total), axis=1)

    # 计算误差
    for i in range(0, 21):
        mse = stats.mean_squared_error(list(expect_future[:, i]), list(predict_future[:, i]))
        nrmse = stats.normalized_rmse(list(expect_future[:, i]), list(predict_future[:, i]))
        mape = stats.mape(list(expect_future[:, i]), list(predict_future[:, i]))
        print "zone", i+1, ":", "mse:", mse, "  nrmse:", nrmse, "  mape:", mape

    # 可视化制图
    plot.plot_comparison(expect_future, predict_future, plot_num=21, algorithm="(XGBoost)")


if __name__ == "__main__":
    main()