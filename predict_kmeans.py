# -*- coding:utf-8 -*-
import load_data
import plot
import statistics as stats
import numpy as np
from scipy.stats import linregress
from scipy.spatial import distance
from scipy.cluster.vq import kmeans

'''
使用k-Means聚类算法进行预测
'''


# 对用电负荷数据进行预处理
def pre_processing_data(series):
    x_data = []
    y_data = []
    for x in range(len(series)-1):
        x_data.append(series[x])
        y_data.append(np.mean(series[x+1]))

    # 将2008/06/01-2008/06/30用电负荷数据作为测试数据
    cut_off = len(series)-30*24
    x_train = x_data[0:cut_off]
    y_train = y_data[0:cut_off]
    x_test = x_data[cut_off:]
    y_test = y_data[cut_off:]

    x_train = [np.log(x+1) for x in x_train]
    x_test = [np.log(x+1) for x in x_test]
    y_train = [np.log(y+1) for y in y_train]

    # 消除时间序列的趋势项
    indices = np.arange(len(y_data))
    train_indices = indices[0:cut_off]
    test_indices = indices[cut_off:]
    regress_result = linregress(train_indices, y_train)
    slope = regress_result[0]
    intercept = regress_result[1]
    detrend = []
    for val in range(len(y_train)):
        pred = slope*val + intercept
        residual = y_train[val] - pred
        detrend.append(residual)
    y_train = detrend
    return x_train, y_train, x_test, y_test, test_indices, slope, intercept


# 定义K-Means聚类算法
def kmeans_clustering(x, k):
    data = np.asarray(x)
    # 计算质心
    centroids = kmeans(data, k, iter=20)[0]
    # 重计算标签
    labels = []
    for y in range(len(x)):
        min_dist = float('inf')
        min_label = -1
        for z in range(len(centroids)):
            e = distance.euclidean(data[y], centroids[z])
            if e < min_dist:
                min_dist = e
                min_label = z
        labels.append(min_label)
    return centroids, labels


# 定义基于聚类的预测算法
def predict_clustering(clusters, cluster_sets, x_test):
    cluster_labels = []
    for i in range(len(x_test)):
        clust_dex = -1
        clust_dist = float('inf')
        for y in range(len(clusters)):
            dist = distance.euclidean(clusters[y], x_test[i])
            if dist < clust_dist:
                clust_dist = dist
                clust_dex = y
        cluster_labels.append(clust_dex)

    # 根据聚类结果进行分类预测
    predict = np.zeros(len(x_test))
    for i in range(len(x_test)):
        x = x_test[i]
        examples = cluster_sets[cluster_labels[i]]
        pred = 0.0
        normalizer = 0.0
        for example in examples:
            if distance.euclidean(x, example[0])==0:
                similarity = 10000.0
            else:
                similarity = 1.0/distance.euclidean(x, example[0])
            pred += similarity * example[1]
            normalizer += similarity
        predict[i] = pred / normalizer
    return predict


# 对每个区域用电负荷进行预处理与预测
def clustering(zone_id, series):
    print "zone", zone_id, ":"
    x_train, y_train, x_test, y_test, test_indices, slope, intercept = pre_processing_data(series)
    # 计算质心和标签
    centroids_k_7, labels_k_7 = kmeans_clustering(x_train, 7)
    centroids_k_24, labels_k_24 = kmeans_clustering(x_train, 24)
    centroids = [centroids_k_7, centroids_k_24]
    labels = [labels_k_7, labels_k_24]

    alg_names = ["(K-Means(7))", "(K-Means(24))"]

    for i in range(len(centroids)):
        centroid = centroids[i]
        label = labels[i]
        cluster_sets = []
        for x in range(len(centroid)):
            cluster_sets.append([])
        for x in range(len(label)):
            cluster_sets[label[x]].append((x_train[x], y_train[x]))

        predict = predict_clustering(centroid, cluster_sets, x_test)

        trend_predict = np.zeros(len(predict))
        for j in range(len(predict)):
            trend_predict[j] = predict[j] + (slope*test_indices[j] + intercept)
        trend_predict = [np.exp(x)-1 for x in trend_predict]

        predicts = trend_predict
        # 计算误差
        mse = stats.mean_squared_error(y_test, trend_predict)
        nrmse = stats.normalized_rmse(y_test, trend_predict)
        mape = stats.mape(y_test, trend_predict)
        print alg_names[i], ":", "mse:", mse, "  nrmse:", nrmse, "  mape:", mape

        plot.plot_comparison(y_test, predicts, plot_num=1, algorithm=alg_names[i], zone_id=zone_id)


def main():
    # 读取用电负荷数据集
    length_load, time_load, load = load_data.read_load_history(is_full=True)
    load = load.reshape(len(load), 20)

    # 对每个区域用电负荷进行预处理与预测
    for zone_id in range(1, 21):
        zone_load_series = load[:, zone_id-1]
        clustering(zone_id, zone_load_series)


if __name__ == "__main__":
    main()