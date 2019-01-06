# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 7:14 PM
# @Author  : WittonZhou
# @File    : EM.py

import numpy as np
import random
import math
import time
'''
EM算法
EM算法的一个重要应用是高斯混合模型的参数估计
----------
数据集：随机生成（高斯分布），设置两个分模型
'''


def load_data(mean0, var0, mean1, var1, alpha0, alpha1):
    """
    初始化数据集，通过服从高斯分布的随机函数来生成数据集
    :param mean0: 高斯0的均值
    :param var0: 高斯0的方差
    :param mean1: 高斯1的均值
    :param var1: 高斯1的方差
    :param alpha0: 高斯0的系数
    :param alpha1: 高斯1的系数
    :return: 混合了两个高斯分布的数据
    """
    data_length = 1000
    data_0 = np.random.normal(mean0, var0, int(data_length * alpha0))
    data_1 = np.random.normal(mean1, var1, int(data_length * alpha1))
    dataset = []
    dataset.extend(data_0)
    dataset.extend(data_1)
    random.shuffle(dataset)
    return dataset


def cal_Gauss(dataset, mean, var):
    """
    根据高斯密度函数计算值
    :param dataset: 可观测数据集
    :param mean: 均值
    :param var: 方差
    :return: 整个可观测数据集的高斯分布密度（向量形式）
    """
    # 计算过程就是依据式9.25写的
    result = (1 / (math.sqrt(2 * math.pi) * var ** 2)) * \
             np.exp(-1 * (dataset - mean) * (dataset - mean) / (2 * var ** 2))
    # 返回结果
    return result


def E_step(dataset, alpha0, mean0, var0, alpha1, mean1, var1):
    """
    EM算法中的E步, 依据当前模型参数，计算分模型k对观数据y的响应度
    :param dataset: 可观测数据y
    :param alpha0: 高斯模型0的系数
    :param mean0: 高斯模型0的均值
    :param var0: 高斯模型0的方差
    :param alpha1: 高斯模型1的系数
    :param mean1: 高斯模型1的均值
    :param var1: 高斯模型1的方差
    :return: 两个模型各自的响应度
    """
    # 计算y0的响应度
    # 先计算模型0的响应度的分子
    gamma0 = alpha0 * cal_Gauss(dataset, mean0, var0)
    # 模型1响应度的分子
    gamma1 = alpha1 * cal_Gauss(dataset, mean1, var1)

    # 两者相加为E步中的分布
    sum = gamma0 + gamma1
    # 各自相除，得到两个模型的响应度
    gamma0 = gamma0 / sum
    gamma1 = gamma1 / sum

    # 返回两个模型响应度
    return gamma0, gamma1


def M_step(mean0, mean1, gamma0, gamma1, dataset):
    """
    EM算法中的M步，依据算法9.2计算各个值
    :param mean0: 高斯模型0的均值
    :param mean1: 高斯模型1的均值
    :param gamma0: 响应度0
    :param gamma1: 响应度1
    :param dataset: 
    :return: 
    """
    mean0_new = np.dot(gamma0, dataset) / np.sum(gamma0)
    mean1_new = np.dot(gamma1, dataset) / np.sum(gamma1)

    sigmod0_new = math.sqrt(np.dot(gamma0, (dataset - mean0) ** 2) / np.sum(gamma0))
    sigmod1_new = math.sqrt(np.dot(gamma1, (dataset - mean1) ** 2) / np.sum(gamma1))

    alpha0_new = np.sum(gamma0) / len(gamma0)
    alpha1_new = np.sum(gamma1) / len(gamma1)

    # 将更新的值返回
    return mean0_new, mean1_new, sigmod0_new, sigmod1_new, alpha0_new, alpha1_new


def EM_train(dataset, iter=10000):
    """
    根据EM算法进行参数估计
    :param dataset: 
    :param iter: 
    :return: 估计的参数
    """
    dataset = np.array(dataset)

    # 步骤1：对参数取初值，开始迭代
    alpha0 = 0.5
    mean0 = 0
    var0 = 1
    alpha1 = 0.5
    mean1 = 1
    var1 = 1

    # 开始迭代
    iter_step = 0
    while (iter_step < iter):
        iter_step += 1
        # 步骤2：E步：依据当前模型参数，计算分模型k对观测数据y的响应度
        gamma0, gamma1 = E_step(dataset, alpha0, mean0, var0, alpha1, mean1, var1)
        # 步骤3：M步：计算新一轮迭代的模型参数
        mean0, mean1, var0, var1, alpha0, alpha1 = M_step(mean0, mean1, gamma0, gamma1, dataset)

    # 迭代结束后将更新后的各参数返回
    return alpha0, mean0, var0, alpha1, mean1, var1

if __name__ == '__main__':
    start = time.time()
    # 两个alpha的和必须为1
    alpha0 = 0.3
    mean0 = -2
    var0 = 0.5
    alpha1 = 0.7
    mean1 = 0.5
    var1 = 1

    dataset = load_data(mean0, var0, mean1, var1, alpha0, alpha1)

    print('目前的参数是：')
    print('alpha0:%.1f, mean0:%.1f, var0:%.1f, alpha1:%.1f, mean1:%.1f, var1:%.1f' % (
        alpha0, mean0, var0, alpha1, mean1, var1))

    # EM训练
    alpha0, mean0, var0, alpha1, mean1, var1 = EM_train(dataset)

    print('现在的参数是：')
    print('alpha0:%.1f, mean0:%.1f, var0:%.1f, alpha1:%.1f, mean1:%.1f, var1:%.1f' % (
        alpha0, mean0, var0, alpha1, mean1, var1))
    print('消耗时间为：', time.time() - start)

