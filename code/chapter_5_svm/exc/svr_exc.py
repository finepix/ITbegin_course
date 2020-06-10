#!/usr/bin/env python

# encoding: utf-8
"""
# @Time    : 2020/6/10
# @Author  : shawn_zhu
# @Site    : 
# @File    : svr_exc.py
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
    return datasets.load_boston()


if __name__ == '__main__':
    boston = load_dataset()

    # 划分训练测试集
    X = boston.data
    Y = boston.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # step1: 训练
    rbf_svr_model = svm.SVR(kernel='rbf', C=1.0, epsilon=0.2)
    rbf_svr_model.fit(X_train, Y_train)

    # step2: 测试
    prediction = rbf_svr_model.predict(X_test)

    # 打印分类结果
    print('预测结果：{}'.format(prediction))

    # 计算准确度等度量
    loss_mae = metrics.mean_absolute_error(y_true=Y_test, y_pred=prediction)
    loss_mse = metrics.mean_squared_error(y_true=Y_test, y_pred=prediction)
    print('绝对误差：%.4f' % loss_mae)
    print('均方误差：%.4f' % loss_mse)