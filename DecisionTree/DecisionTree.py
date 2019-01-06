# -*- coding: utf-8 -*-
# @Time    : 2019/1/2 1:47 PM
# @Author  : WittonZhou
# @File    : DecisionTree.py

import numpy as np
import time

'''
决策树模型
----------
数据集：Minst
分类结果：
准确率(未剪枝)：0.8589
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
            # 二值化处理，大于128的转换成1，小于的转换成0
            data.append([int(int(num) > 128) for num in current_line[1:]])
            label.append(int(current_line[0]))

    return data, label


def majority_class(label):
    """
    返回当前标签集中所占数目最多的类别
    :param label: 
    :return: 
    """
    class_dict = {}
    for i in range(len(label)):
        if label[i] in class_dict.keys():
            class_dict[label[i]] += 1
        else:
            class_dict[label[i]] = 1
    class_sort = sorted(class_dict.items(), key=lambda x: x[1], reverse=True)
    return class_sort[0][0]


def cal_H_D(train_label):
    """
    计算数据集D的经验熵
    :param train_label: 
    :return: 
    """
    H_D = 0
    train_label_set = set([label for label in train_label])
    for i in train_label_set:
        p = train_label[train_label == i].size / train_label.size
        H_D += -1 * p * np.log2(p)
    return H_D


def cal_H_D_A(train_data_dev_feature, train_label):
    """
    计算经验条件熵
    :param train_data_dev_feature: 分割后只有feature那列数据的数组
    :param train_label: 
    :return: 
    """
    H_D_A = 0
    train_data_set = set([label for label in train_data_dev_feature])
    for i in train_data_set:
        H_D_A += train_data_dev_feature[train_data_dev_feature == i].size / train_data_dev_feature.size \
                 * cal_H_D(train_label[train_data_dev_feature == i])
    return H_D_A


def select_best_feature(train_data, train_label):
    """
    根据最大信息增益，选择最优的划分数据集的特征
    :param train_data: 
    :param train_label: 
    :return: 
    """
    train_data = np.array(train_data)
    train_label = np.array(train_label).T
    feature_num = train_data.shape[1]
    # 初始化最大增益好和最优特征
    max_gain = -1
    best_feature = -1
    for feature in range(feature_num):
        # 1.计算数据集D的经验熵H(D)
        H_D = cal_H_D(train_label)
        # 2.计算条件经验熵H(D|A)
        train_data_dev_feature = np.array(train_data[:, feature].flat)
        # 3.计算信息增益G(D|A)    G(D|A) = H(D) - H(D|A)
        gain = H_D - cal_H_D_A(train_data_dev_feature, train_label)
        # 更新最大增益和最优特征
        if gain > max_gain:
            max_gain = gain
            best_feature = feature

    return best_feature, max_gain


def get_sub_data(train_data, train_label, A, a):
    """
    获得划分完的子集
    :param train_data: 
    :param train_label: 
    :param A: 划分的特征索引
    :param a: 当data[A]== a时，说明该行样本需要被保留
    :return: 
    """
    return_data = []
    return_label = []
    for i in range(len(train_data)):
        if train_data[i][A] == a:
            return_data.append(train_data[i][0:A] + train_data[i][A+1:])
            return_label.append(train_label[i])

    return return_data, return_label


def create_tree(*dataset):
    """
    递归构建决策树
    :param dataset: 
    :return: 
    """
    # 初始化信息增益的阈值
    Epsilon = 0.1
    # 之所以使用元祖作为参数，是由于后续递归调用时直数据集需要对某个特征进行切割，
    # 在函数递归调用上直接将切割函数的返回值放入递归调用中，而函数的返回值形式是元祖的。
    train_data = dataset[0][0]
    train_label = dataset[0][1]

    print('start a node', len(train_data[0]), len(train_label))

    class_dict = {i for i in train_label}

    # 如果D中所有实例属于同一类Ck，则置T为单节点数，并将Ck作为该节点的类，返回T
    if len(class_dict) == 1:
        return train_label[0]
    # 如果A为空集，则置T为单节点数，并将D中实例数最大的类Ck作为该节点的类，返回T
    if len(train_data[0]) == 0:
        return majority_class(train_label)
    # 否则，按式5.10计算A中个特征值的信息增益，选择信息增益最大的特征Ag
    Ag, EpsilonGet = select_best_feature(train_data, train_label)
    # 如果Ag的信息增益比小于阈值Epsilon，则置T为单节点树，并将D中实例数最大的类Ck作为该节点的类，返回T
    if EpsilonGet < Epsilon:
        return majority_class(train_label)

    # 否则，对Ag的每一可能值ai，依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的
    # 类作为标记，构建子节点，由节点及其子节点构成树T，返回T
    tree_dict = {Ag: {}}
    # 特征值为0时，进入0分支
    # getSubDataArr(trainDataList, trainLabelList, Ag, 0)：在当前数据集中切割当前feature，返回新的数据集和标签集
    tree_dict[Ag][0] = create_tree(get_sub_data(train_data, train_label, Ag, 0))
    tree_dict[Ag][1] = create_tree(get_sub_data(train_data, train_label, Ag, 1))

    return tree_dict


def predict(test_data, tree):
    """
    预测类别
    :param test_data: 测试数据集
    :param tree: 决策树
    :return: 预测类别结果
    """
    while True:
        (key, value), = tree.items()
        # 如果当前的value是字典，说明还需要遍历下去
        if type(tree[key]).__name__ == 'dict':
            data_value = test_data[key]
            del test_data[key]
            tree = value[data_value]
            if type(tree).__name__ == 'int':
                return tree
        else:
            return tree


def test(test_data, test_label, tree):
    """
    测试test数据，返回准确率
    :param test_data: 测试数据集
    :param test_label: 测试数据标签
    :param tree: 训练集生成的树
    :return: 准确率
    """
    error_count = 0
    for i in range(len(test_data)):
        if test_label[i] != predict(test_data[i], tree):
            error_count += 1

    return 1 - error_count / len(test_data)


if __name__ == '__main__':
    start = time.time()
    print('starting to load data')
    train_data, train_label = load_data('../Input/mnist_train.csv')
    test_data, test_label = load_data('../Input/mnist_test.csv')

    # 创建决策树
    print('starting to create tree')
    tree = create_tree((train_data, train_label))
    print('tree is:', tree)

    # 测试准确率
    print('starting to test')
    accuracy = test(test_data, test_label, tree)
    print('准确率为：', accuracy)
    print('消耗时间为：', time.time() - start)