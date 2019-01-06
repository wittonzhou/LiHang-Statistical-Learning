# -*- coding: utf-8 -*-
# @Time    : 2018/12/31 2:20 PM
# @Author  : WittonZhou
# @File    : perceptron.py

import numpy as np
import time
'''
感知机模型
当训练数据集线性可分时，感知机学习算法是收敛的，且存在无穷个解。
其解由于不同的初值或不同的迭代顺序而可能有所不同。
----------
数据集：Minst
分类结果（二分类）：
准确率：0.7802
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
            # Mnsit有0-9是个标记，由于是二分类任务，故将>=5的作为1，<5为-1
            if int(current_line[0]) >= 5:
                label.append(1)
            else:
                label.append(-1)

            data.append([int(num) for num in current_line[1:]])

    return data, label


def perceptron(train_data, train_label, iter=50):
    """
    感知机模型训练过程
    :param train_data: 
    :param train_label: 
    :param iter: 迭代次数
    :return: 
    """
    data_matrix = np.mat(train_data)
    label_matrix = np.mat(train_label).T

    m, n = np.shape(data_matrix)
    # 初始化W, b, learning_rate
    W = np.zeros((1, np.shape(data_matrix)[1]))
    b = 0
    learning_rate = 0.0001
    for k in range(iter):
        for i in range(m):
            Xi = data_matrix[i]
            yi = label_matrix[i]
            # 对于在该超平面先的误分类点，更新W和b
            if -1 * yi * (W * Xi.T + b) >= 0:
                W = W + learning_rate * yi * Xi
                b = b + learning_rate * yi
        print('Round %d: %d training' % (k, iter))
    return W, b


def test(test_data, test_label, W, b):
    """
    感知机模型测试过程
    :param test_data: 测试数据集
    :param test_label: 测试标签
    :param W: 权值
    :param b: 偏置
    :return: 准确率
    """
    # 将数据集转换为矩阵形式方便运算
    data_matrix = np.mat(test_data)
    label_matrix = np.mat(test_label).T
    m, n = np.shape(data_matrix)
    error_count = 0
    for i in range(m):
        Xi = data_matrix[i]
        yi = label_matrix[i]
        result = yi * (W * Xi.T + b)
        if result < 0:
            error_count += 1

    accuary = 1 - (error_count / m)
    return accuary

if __name__ == '__main__':
    start = time.time()
    print('starting to load data')
    train_data, train_label = load_data('../Input/mnist_train.csv')
    test_data, test_label = load_data('../Input/mnist_test.csv')

    print('starting to train')
    W, b = perceptron(train_data, train_label, iter=100)

    print('starting to test')
    accuracy = test(test_data, test_label, W, b)

    print('准确率为：', accuracy)
    print('消耗时间为：', time.time() - start)


