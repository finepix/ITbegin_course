#!/usr/bin/env python

# encoding: utf-8
"""
# @Time    : 2020/6/10
# @Author  : shawn_zhu
# @Site    : 
# @File    : svm_exc.py
# @Software: PyCharm

"""


from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn import metrics


def load_dataset():
    """
            加载数据
    :return:
    """
    return datasets.load_iris()


if __name__ == '__main__':
    iris = load_dataset()

    # 划分训练测试集
    X = iris.data
    Y = iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # step1: 训练
    rbf_svm_model = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)
    rbf_svm_model.fit(X_train, Y_train)

    # step2: 测试
    prediction = rbf_svm_model.predict(X_test)

    # 打印分类结果
    print('预测结果：{}'.format(prediction))

    # 计算准确度等度量
    results = metrics.accuracy_score(Y_test, prediction)
    print('测试准确率：%.4f' % results)