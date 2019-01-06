# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 10:34 AM
# @Author  : WittonZhou
# @File    : AdaBoost.py

import numpy as np
import time
'''
AdaBoost模型
----------
数据集：Minst
分类结果（二分类）：
准确率：0.978
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
            # 二值化处理，大于128的转换成1，小于的转换成0。
            data.append([int(int(num) > 128) for num in current_line[1:]])
            if int(current_line[0]) == 0:
                label.append(1)
            else:
                label.append(-1)
        return data, label


def cal_e_Gx(train_data, train_label, n, div, rule, D):
    """
    计算分类错误率
    :param train_data: 
    :param train_label: 
    :param n: 要操作的特征
    :param div: 划分点
    :param rule: 正反例标签
    :param D: 权值分布D
    :return: 预测结果， 分类误差率
    """
    # 初始化分类误差率为0
    e = 0
    # 将训练数据矩阵中特征为n的那一列单独剥出来做成数组
    x = train_data[:, n]
    # 同样将标签也转换成数组格式，x和y的转换只是单纯为了提高运行速度
    # 测试过相对直接操作而言性能提升很大
    y = train_label
    predict = []

    # 依据小于和大于的标签依据实际情况会不同，在这里直接进行设置
    if rule == 'LisOne':
        L = 1
        H = -1
    else:
        L = -1
        H = 1

    # 遍历所有样本的特征m
    for i in range(train_data.shape[0]):
        if x[i] < div:
            # 如果小于划分点，则预测为L
            predict.append(L)
            # 如果预测错误，分类错误率要加上该分错的样本的权值
            if y[i] != L:
                e += D[i]
        elif x[i] >= div:
            predict.append(H)
            if y[i] != H:
                e += D[i]
    # 返回预测结果和分类错误率e
    return np.array(predict), e


def create_single_boosting_tree(train_data, train_label, D):
    """
    创建单棵提升树
    :param train_data: 
    :param train_label: 
    :param D: 算法8.1中的D，即权值分布
    :return: 创建的单棵提升树
    """
    # 获得样本数目，特征数
    m, n = np.shape(train_data)
    # 单层树的字典，用于存放当前层提升树的参数
    # 也可以认为该字典代表了一层提升树
    single_boost_tree = {}
    # 初始化分类误差率，分类误差率在算法8.1步骤（2）（b）有提到误差率最高也只能100%，因此初始化为1
    single_boost_tree['e'] = 1

    # 对每一个特征进行遍历，寻找用于划分的最合适的特征
    for i in range(n):
        # 因为特征已经经过二值化，只能为0和1，因此分切分时分为-0.5， 0.5， 1.5三挡进行切割
        for div in [-0.5, 0.5, 1.5]:
            # 在单个特征内对正反例进行划分时，有两种情况：
            # 可能是小于某值的为1，大于某值得为-1，也可能小于某值得是-1，反之为1
            # 因此在寻找最佳提升树的同时对于两种情况也需要遍历运行
            # LisOne：Low is one：小于某值得是1
            # HisOne：High is one：大于某值得是1
            for rule in ['LisOne', 'HisOne']:
                # 按照第i个特征，以值div进行切割，进行当前设置得到的预测和分类错误率
                Gx, e = cal_e_Gx(train_data, train_label, i, div, rule, D)
                # 如果分类错误率e小于当前最小的e，那么将它作为最小的分类错误率保存
                if e < single_boost_tree['e']:
                    single_boost_tree['e'] = e
                    # 同时也需要存储最优划分点、划分规则、预测结果、特征索引
                    # 以便进行D更新和后续预测使用
                    single_boost_tree['div'] = div
                    single_boost_tree['rule'] = rule
                    single_boost_tree['Gx'] = Gx
                    single_boost_tree['feature'] = i
    # 返回单层的提升树
    return single_boost_tree


def create_boosting_tree(train_data, train_label, tree_num = 100):
    """
    创建提升树
    :param train_data: 
    :param train_label: 
    :param tree_num: 提升树的轮数
    :return: 提升树
    """
    # 将数据和标签转化为数组形式
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    # 每增加一轮后，当前最终预测结果列表
    final_predict = [0] * len(train_label)
    # 获得训练集数量，特征数
    m, n = np.shape(train_data)

    # 依据算法8.1步骤（1）初始化D为1/N，相当于初始化每个样本的权重一致
    D = [1 / m] * m
    # 初始化提升树列表
    tree = []
    # 循环创建提升树
    for i in range(tree_num):
        # 得到当前层的提升树
        current_tree = create_single_boosting_tree(train_data, train_label, D)
        # 根据式8.2计算当前层的alpha
        alpha = 1 / 2 * np.log((1 - current_tree['e']) / current_tree['e'])
        # 获得当前层的预测结果，用于下一步更新D
        Gx = current_tree['Gx']
        # 依据式8.4更新D
        # 考虑到该式每次只更新D中的一个w，要循环进行更新知道所有w更新结束会很复杂，
        # 所以该式以向量相乘的形式，一个式子将所有w全部更新完。
        # np.multiply(trainLabelArr, Gx)：exp中的y*Gm(x)，结果是一个行向量，内部为yi*Gm(xi)
        # np.exp(-1 * alpha * np.multiply(trainLabelArr, Gx))：上面求出来的行向量内部全体成员再乘以-αm，
        # 然后取对数，和书上式子一样，只不过书上式子内是一个数，这里是一个向量。
        D = np.multiply(D, np.exp(-1 * alpha * np.multiply(train_label, Gx))) / sum(D)
        # 在当前层参数中增加alpha参数，预测的时候需要用到
        current_tree['alpha'] = alpha
        # 将当前层添加到提升树索引中。
        tree.append(current_tree)

        # 根据8.6式将结果加上当前层乘以α，得到目前的最终输出预测
        final_predict += alpha * Gx
        # 计算当前最终预测输出与实际标签之间的误差
        error = sum([1 for i in range(len(train_data)) if np.sign(final_predict[i]) != train_label[i]])
        # 计算当前最终误差率
        final_error = error / len(train_data)
        # 如果误差为0，提前退出即可，因为没有必要再计算算了
        if final_error == 0:
            return tree
        # 打印信息
        print('iter:%d:%d, sigle error:%.4f, finall error:%.4f' % (i, tree_num, current_tree['e'], final_error))
    # 返回整个提升树
    return tree


def predict(x, div, rule, feature):
    """
    输出预测结果
    :param x: 预测样本
    :param div: 划分点
    :param rule: 划分规则
    :param feature: 进行操作的特征
    :return: 
    """
    if rule == 'LisOne':
        L = 1
        H = -1
    else:
        L = -1
        H = 1

    # 判断预测结果
    if x[feature] < div:
        return L
    else:
        return H


def test(test_data, test_label, tree):
    """
    测试
    :param test_data: 
    :param test_label: 
    :param tree: 
    :return: 准确率
    """
    error_count = 0
    # 遍历每一个测试样本
    for i in range(len(test_data)):
        # 预测结果值，初始为0
        result = 0
        # 依据算法8.1式8.6
        # 预测式子是一个求和式，对于每一层的结果都要进行一次累加
        # 遍历每层的树
        for current_tree in tree:
            # 获取该层参数
            div = current_tree['div']
            rule = current_tree['rule']
            feature = current_tree['feature']
            alpha = current_tree['alpha']
            # 将当前层结果加入预测中
            result += alpha * predict(test_data[i], div, rule, feature)
        # 预测结果取sign值（指示函数）
        if np.sign(result) != test_label[i]:
            error_count += 1
    return 1 - error_count / len(test_data)


if __name__ == '__main__':
    start = time.time()
    print('starting to load data')
    train_data, train_label = load_data('../Input/mnist_train.csv')
    test_data, test_label = load_data('../Input/mnist_test.csv')
    # 创建提升树
    print('starting to train')
    tree = create_boosting_tree(train_data[:10000], train_label[:10000], 50)

    print('starting to test')
    accuracy = test(test_data[:1000], test_label[:1000], tree)

    print('准确率为：', accuracy)
    print('消耗时间为：', time.time() - start)
