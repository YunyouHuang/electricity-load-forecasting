# -*- coding:utf-8 -*-
import math
'''
计算误差
'''


# 计算Mean Squared Error
def mean_squared_error(actual, predict):
    if not len(actual) == len(predict):
        return -1.0
    s = 0.0
    for x in range(len(actual)):
        s += math.pow(actual[x] - predict[x], 2)
    return s/len(actual)


# 计算Normalized Root Mean Square Error(NRMSE)
def normalized_rmse(actual, predict):
    if not len(actual) == len(predict):
        return -1.0
    sum_squares = 0.0
    max_y = actual[0]
    min_y = actual[0]
    for x in range(len(actual)):
        sum_squares += math.pow(predict[x]-actual[x], 2)
        max_y = max(max_y, actual[x])
        min_y = min(min_y, actual[x])
    return math.sqrt(sum_squares/len(actual))/(max_y-min_y)


# 计算Mean Absolute Percent Error(MAPE)
def mape(actual, predict):
    if not len(actual) == len(predict):
        return -1.0
    s = 0.0
    for x in range(len(actual)):
        s += abs((actual[x]-predict[x])/actual[x])
    return s/len(actual)
