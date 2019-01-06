# -*- coding: utf-8 -*-
# @Time    : 2018/12/31 2:58 PM
# @Author  : WittonZhou
# @File    : knn.py

import numpy as np
import time

'''
K近邻模型
----------
数据集：Minst
分类结果：
准确率：0.99
'''


def load_data(filename):
    """
    加载数据集
    :param filename: 
    :return: 
    """
    data = []
    label = []
    with open(filename) as f:
        for line in f.readlines():
            current_line = line.strip().split(',')
            data.append([int(num) for num in current_line[1:]])
            label.append(int(current_line[0]))

    return data, label


def calculate_distance(x1, x2):
    """
    计算欧式距离
    :param x1: 
    :param x2: 
    :return: 两点的欧式距离
    """
    return np.sqrt(np.sum(np.square(x1 - x2)))


def getKNeighbor(data_matrix, label_matrix, x, K):
    """
    获得K个近邻，返回近邻中最多的类别标记
    :param data_matrix: 
    :param label_matrix: 
    :param x: 
    :param K: 
    :return: 
    """
    distList = [0] * len(label_matrix)

    for i in range(len(data_matrix)):
        x1 = data_matrix[i]
        current_dist = calculate_distance(x1, x)
        distList[i] = current_dist

    topKList = np.argsort(np.array(distList))[:K]
    # 分类决策使用的是投票表决，返回K个近邻中label最多的标记
    labelList = [0] * 10
    for index in topKList:
        labelList[int(label_matrix[index])] += 1
    return labelList.index(max(labelList))


def test(train_data, train_label, test_data, test_label, K):
    """
    KNN没有显式的学习过程
    :param train_data: 
    :param train_label: 
    :param test_data: 
    :param test_label: 
    :param K: 
    :return: 
    """
    train_data_matrix = np.mat(train_data)
    train_label_matrix = np.mat(train_label).T
    test_data_matrix = np.mat(test_data)
    test_label_matrix = np.mat(test_label).T

    error_count = 0
    # 过于费时，仅选择100条数据来测试结果
    for i in range(100):
        print('test %d:%d' % (i, 100))
        x = test_data_matrix[i]
        y = getKNeighbor(train_data_matrix, train_label_matrix, x, K)
        if y != test_label_matrix[i]:
            error_count += 1

    return 1 - (error_count / 100)


if __name__ == '__main__':
    start = time.time()

    print('starting to load data')
    train_data, train_label = load_data('../Input/mnist_train.csv')
    test_data, test_label = load_data('../Input/mnist_test.csv')

    print('starting to test')
    accuracy = test(train_data, train_label, test_data, test_label, 5)
    print('准确率为：', accuracy)
    print('消耗时间为：', time.time() - start)

