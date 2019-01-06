# -*- coding: utf-8 -*-
# @Time    : 2019/1/2 12:44 PM
# @Author  : WittonZhou
# @File    : NaiveBayes.py

import numpy as np
import time

'''
朴素贝叶斯模型
----------
数据集：Minst
分类结果：
准确率：0.843
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
            # 对数据进行了二值化处理，大于128的转换成1，小于的转换成0
            data.append([int(int(num) > 128) for num in current_line[1:]])
            label.append(int(current_line[0]))

    return data, label


def NaiveBayes(Py, Px_y, x):
    """
    通过朴素贝叶斯进行概率估计
    :param Py: 先验概率分布
    :param Px_y: 条件概率分布
    :param x: 
    :return: 估计概率最高的label
    """
    # 设置特征数目
    feature_num = 784
    # 设置类别数目
    class_num = 10

    P = [0] * class_num
    for i in range(class_num):
        sum = 0
        for j in range(feature_num):
            # log转换后，连乘变为相加
            sum += Px_y[i][j][x[j]]
        # log转换后，连乘变为相加
        P[i] = sum + Py[i]
    # 找到后验概率最大值对应的索引
    return P.index(max(P))


def test(Py, Px_y, test_data, test_label):
    """
    使用学习得到的先验概率分布和条件概率分布对测试集进行测试，返回准确率
    :param Py: 先验概率分布
    :param Px_y: 条件概率分布
    :param test_data: 测试集数据
    :param test_label: 测试集标记
    :return: 准确率
    """
    error_count = 0
    for i in range(len(test_data)):
        predict = NaiveBayes(Py, Px_y, test_data[i])
        if predict != test_label[i]:
            error_count += 1
    return 1 - (error_count / len(test_data))


def get_all_prob(train_data, train_label):
    """
    通过训练集计算先验概率分布和条件概率分布
    :param train_data: 
    :param train_label: 
    :return: 训练集的先验概率分布和条件概率分布
    """
    # 初始化特征数和类别数
    feature_num = 784
    class_num = 10

    Py = np.zeros((class_num, 1))
    for i in range(class_num):
        # 分子加1，分母加上class_num来进行平滑处理
        Py[i] = ((np.sum(np.mat(train_label) == i)) + 1) / (len(train_label) + 10)

    # 连乘项通过log以后，可以变成各项累加，简化计算
    Py = np.log(Py)

    # Px_y = P（X = x | Y = y）
    # 初始化为全0矩阵，用于存放所有情况下的条件概率
    Px_y = np.zeros((class_num, feature_num, 2))
    for i in range(len(train_label)):
        label = train_label[i]
        x = train_data[i]
        for j in range(feature_num):
            # 在矩阵中对应位置加1，先把所有数累加，
            Px_y[label][j][x[j]] += 1

    for label in range(class_num):
        for j in range(feature_num):
            # 获取y=label，第j个特诊为0的个数
            Px_y0 = Px_y[label][j][0]
            # 获取y=label，第j个特诊为1的个数
            Px_y1 = Px_y[label][j][1]
            # 分别计算对于y= label，x第j个特征为0和1的条件概率分布
            # 这里同样进行平滑处理
            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))

    return Py, Px_y

if __name__ == '__main__':
    start = time.time()

    print('starting to load data')
    train_data, train_label = load_data('../Input/mnist_train.csv')
    test_data, test_label = load_data('../Input/mnist_test.csv')

    print('starting to train')
    Py, Px_y = get_all_prob(train_data, train_label)

    print('starting to test')
    accuracy = test(Py, Px_y, test_data, test_label)

    print('准确率为：', accuracy)
    print('消耗时间为：', time.time() - start)