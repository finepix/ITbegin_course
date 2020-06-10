#!/usr/bin/env python

# encoding: utf-8
"""
# @Time    : 2020/6/10
# @Author  : shawn_zhu
# @Site    : 
# @File    : spectral_clustering_exc.py
# @Software: PyCharm

"""

import numpy as np

from sklearn import datasets, metrics
from sklearn.cluster import SpectralClustering

def load_dataset():
    return datasets.load_wine()


if __name__ == '__main__':
    # 加载数据集
    wine = load_dataset()
    X = wine.data[:, :2]
    Y = wine.target
    n_classes = np.unique(Y).shape[0]

    # 初始化以及训练模型(尝试多种gamma参数，或者多种关联矩阵度量)
    spectral_clustering_model = SpectralClustering(n_clusters=n_classes, affinity='rbf', gamma=0.2, random_state=4399)
    prediction = spectral_clustering_model.fit_predict(X)

    # 数据聚类结果以及聚类评价指标
    mutual_info = metrics.cluster.mutual_info_score(labels_true=Y, labels_pred=prediction)

    print('互信息：%.4f' % mutual_info)
    print('模型：{}'.format(spectral_clustering_model))


