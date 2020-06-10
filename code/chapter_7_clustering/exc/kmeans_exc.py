#!/usr/bin/env python

# encoding: utf-8
"""
# @Time    : 2020/6/10
# @Author  : shawn_zhu
# @Site    : 
# @File    : kmeans_exc.py
# @Software: PyCharm

"""
import numpy as np

from sklearn import datasets, metrics
from sklearn.cluster import KMeans

def load_dataset():
    return datasets.load_wine()


if __name__ == '__main__':
    # 加载数据集
    wine = load_dataset()
    X = wine.data[:, :2]
    Y = wine.target
    n_classes = np.unique(Y).shape[0]

    # 初始化以及训练模型
    kmeans_model = KMeans(n_clusters=n_classes, random_state=4399)
    kmeans_model.fit(X)
    prediction = kmeans_model.predict(X)

    # 数据聚类结果以及聚类评价指标
    mutual_info = metrics.cluster.mutual_info_score(labels_true=Y, labels_pred=prediction)

    print('互信息：%.4f' % mutual_info)
    for idx, center in enumerate(kmeans_model.cluster_centers_):
        print('聚类中心{}：{}'.format(idx+1, center))


