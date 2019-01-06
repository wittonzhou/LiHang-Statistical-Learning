# -*- coding: utf-8 -*-
# @Time    : 2019/1/2 3:29 PM
# @Author  : WittonZhou
# @File    : LogisticRegression.py

import numpy as np
import time

'''
逻辑回归模型
----------
数据集：Minst
分类结果（二分类）：
准确率：0.902
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
            # 由于是二分类任务，故将标记0的作为1，其余为0
            if int(current_line[0]) == 0:
                label.append(1)
            else:
                label.append(0)
            data.append([int(num) for num in current_line[1:]])
    return data, label


def predict(W, x):
    """
    预测
    :param W: 训练过程中学到的W
    :param x: 要预测的样本
    :return: 预测结果
    """
    # 点乘
    Wx = np.dot(W, x)
    sigmoid_P = np.exp(Wx) / (1 + np.exp(Wx))
    # 阈值为0.5
    if sigmoid_P >= 0.5:
        return 1
    return 0


def logistic_regression(train_data, train_label, iter=100):
    """
    逻辑回归训练过程
    :param train_data: 训练集
    :param train_label: 训练标签
    :param iter: 迭代次数
    :return: W
    """
    for i in range(len(train_data)):
        train_data[i].append(1)

    train_data = np.array(train_data)
    # 初始化权值和学习率
    W = np.zeros(train_data.shape[1])
    learning_rate = 0.001
    for i in range(iter):
        for j in range(train_data.shape[0]):
            Wx = np.dot(W, train_data[j])
            Xj = train_data[j]
            yj = train_label[j]
            # W的更新公式，根据对数似然函数得出
            W += learning_rate * (Xj * yj - (np.exp(Wx) * Xj) / (1 + np.exp(Wx)))

    return W


def test(test_data, test_label, W):
    """
    测试学习到的逻辑回归的分类准确率
    :param test_data: 测试数据集
    :param test_label: 真实测试标签
    :param W: 学习到的W
    :return: 准确率
    """
    for i in range(len(test_data)):
        test_data[i].append(1)

    error_count = 0
    for i in range(len(test_data)):
        if test_label[i] != predict(W, test_data[i]):
            error_count += 1

    return 1 - error_count / len(test_data)


if __name__ == '__main__':
    start = time.time()

    print('starting to load data')
    train_data, train_label = load_data('../Input/mnist_train.csv')
    test_data, test_label = load_data('../Input/mnist_test.csv')

    print('starting to train')
    W = logistic_regression(train_data, train_label)

    print('starting to test')
    accuracy = test(test_data, test_label, W)
    print('准确率为：', accuracy)
    print('消耗时间为：', time.time() - start)







