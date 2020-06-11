#!/usr/bin/env python

# encoding: utf-8
"""
# @Time    : 2020/6/11
# @Author  : shawn_zhu
# @Site    : 
# @File    : lle_exc.py
# @Software: PyCharm

"""

import numpy as np
import time

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import LocallyLinearEmbedding


def load_dataset():
    """
            从sklearn加载数据
    :return:
    """
    return datasets.load_digits()

if __name__ == '__main__':

    # 加载数据，并对数据进行划分
    digits = load_dataset()

    X = digits.data
    Y = digits.target
    n_features = X.shape[1]
    n_classes = np.unique(Y).shape[0]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # TODO: 原始空间的分类效果
    origin_time_start = time.time()
    clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    origin_time_end = time.time()

    acc_origin_space = metrics.accuracy_score(Y_test, prediction)
    time_elapse = (origin_time_end - origin_time_start) * 1000
    print('原始空间的准确率：%.4f, 原始空间数据维度:%d， 耗时：%d ms。' % (acc_origin_space,
                                                    n_features, time_elapse))

    # TODO: 使用lda对数据进行降维
    subspace_dim = 56

    lle_model = LocallyLinearEmbedding(n_components=subspace_dim, n_neighbors=5, random_state=4399)
    lle_model.fit(X_train)

    X_train_new = lle_model.transform(X_train)
    X_test_new = lle_model.transform(X_test)

    # TODO: 在子空间上的分类效果
    subspace_time_start = time.time()
    clf_new = KNeighborsClassifier(n_neighbors=5, weights='distance')
    clf_new.fit(X_train_new, Y_train)
    prediction_subspace = clf_new.predict(X_test_new)
    subspace_time_end = time.time()

    acc_subspace_score = metrics.accuracy_score(Y_test, prediction_subspace)
    time_elapse = (subspace_time_end - subspace_time_start) * 1000
    print('子空间的准确率：%.4f, 子空间数据维度：%d， 耗时：%d ms。' % (acc_subspace_score,
                                                   subspace_dim, time_elapse))
