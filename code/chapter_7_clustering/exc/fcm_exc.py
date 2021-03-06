#!/usr/bin/env python

# encoding: utf-8
"""
# @Time    : 2020/6/11
# @Author  : shawn_zhu
# @Site    : 
# @File    : fcm_exc.py
# @Software: PyCharm

"""
import copy
import math
import random
import time
import numpy as np

from sklearn import metrics, datasets


MAX = 10000.0       # 用于初始化隶属度矩阵U

Epsilon = 0.0000001 # 结束条件


def load_dataset():
    """
        加载wine数据
    :return:
    """
    return datasets.load_wine()

def print_matrix(list):
    """
        以可重复的方式打印矩阵
    :param list:
    :return:
    """
    for i in range(0, len(list)):
        print(list[i])


def initialize_U(data, cluster_number):
    """
        这个函数是隶属度矩阵U的每行加起来都为1. 此处需要一个全局变量MAX.
    :param data:
    :param cluster_number:
    :return:
    """
    global MAX
    U = []
    for i in range(0, len(data)):
        current = []
        rand_sum = 0.0
        for j in range(0, cluster_number):
            dummy = random.randint(1, int(MAX))
            current.append(dummy)
            rand_sum += dummy
        for j in range(0, cluster_number):
            current[j] = current[j] / rand_sum
        U.append(current)
    return U


def distance(point, center):
    """
        该函数计算2点之间的距离（作为列表）。我们指欧几里德距离。闵可夫斯基距离
    :param point:
    :param center:
    :return:
    """
    if len(point) != len(center):
        return -1
    dummy = 0.0
    for i in range(0, len(point)):
        dummy += abs(point[i] - center[i]) ** 2
    return math.sqrt(dummy)


def end_conditon(U, U_old):
    """
        结束条件。当U矩阵随着连续迭代停止变化时，触发结束
    :param U:
    :param U_old:
    :return:
    """
    global Epsilon
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > Epsilon:
                return False
    return True


def get_fuzzy_labels(U):
    """
        处理得到结果
    :param U:
    :return:
    """
    ans = list()
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] == maximum:
                ans.append(j)
                break
    return ans


def fuzzy(data, cluster_number, m):
    """
        这是主函数，它将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
        输入参数：簇数(cluster_number)、隶属度的因子(m)的最佳取值范围为[1.5，2.5]
    :param data:
    :param cluster_number:
    :param m:
    :return:
    """
    # 初始化隶属度矩阵U
    U = initialize_U(data, cluster_number)
    # print_matrix(U)
    # 循环更新U
    while (True):
        # 创建它的副本，以检查结束条件
        U_old = copy.deepcopy(U)
        # 计算聚类中心
        C = []
        for j in range(0, cluster_number):
            current_cluster_center = []
            for i in range(0, len(data[0])):
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(0, len(data)):
                    # 分子
                    dummy_sum_num += (U[k][j] ** m) * data[k][i]
                    # 分母
                    dummy_sum_dum += (U[k][j] ** m)
                # 第i列的聚类中心
                current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
            # 第j簇的所有聚类中心
            C.append(current_cluster_center)

        # 创建一个距离向量, 用于计算U矩阵。
        distance_matrix = []
        for i in range(0, len(data)):
            current = []
            for j in range(0, cluster_number):
                current.append(distance(data[i], C[j]))
            distance_matrix.append(current)

        # 更新U
        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0, cluster_number):
                    # 分母
                    dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2 / (m - 1))
                U[i][j] = 1 / dummy

        if end_conditon(U, U_old):
            print("已完成聚类")
            break

    return U


if __name__ == '__main__':

    # 加载数据
    wine = load_dataset()
    X = wine.data
    Y = wine.target
    n_classes = np.unique(Y).shape[0]

    # 调用模糊C均值函数
    start = time.time()
    res_U = fuzzy(X, n_classes, 1.5)
    predictions = get_fuzzy_labels(res_U)
    end = time.time()

    # 观察模糊聚类结果
    temp = res_U[:5][:]
    print('部分模糊聚类结果：')
    print_matrix(temp)

    # 计算互信息等指标
    mutual_info = metrics.cluster.mutual_info_score(labels_true=Y, labels_pred=predictions)
    time_elapse = (end - start) * 1000
    print("互信息：%.4f, 用时：%.2f ms" % (mutual_info, time_elapse))