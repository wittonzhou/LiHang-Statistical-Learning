# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 9:58 AM
# @Author  : WittonZhou
# @File    : SVM.py

import numpy as np
import time
import math
import random
'''
支持向量机模型
----------
数据集：Minst
分类结果（二分类）：
准确率：
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
            # 转换数据类型
            data.append([int(num) / 255 for num in current_line[1:]])
            # 数字0标记为1,其余标记为-1
            if int(current_line[0]) == 0:
                label.append(1)
            else:
                label.append(-1)
    return data, label


class SVM:
    def __init__(self, train_data, train_label, sigma=10, C=200, toler=0.001):
        """
        SVM相关参数初始化
        :param train_data: 训练数据集
        :param train_label: 训练标签
        :param sigma: 高斯核中分母的σ
        :param C: 软间隔中的惩罚参数
        :param toler: 松弛变量
        """
        # 训练数据集
        self.train_data_mat = np.mat(train_data)
        # 训练标签集，转置为列向量
        self.train_label_mat = np.mat(train_label).T
        # m：训练集数    n：特征数
        self.m, self.n = np.shape(self.train_data_mat)
        # 高斯核分母中的σ
        self.sigma = sigma
        # 惩罚参数
        self.C = C
        # 松弛变量
        self.toler = toler
        # 核函数（初始化时提前计算）
        self.k = self.cal_kernel()
        # SVM中的偏置b
        self.b = 0
        # α 长度为训练集数目
        self.alpha = [0] * self.train_data_mat.shape[0]
        # SMO运算过程中的Ei
        self.E = [0 * self.train_label_mat[i, 0] for i in range(self.train_label_mat.shape[0])]
        self.supportVecIndex = []

    def cal_kernel(self):
        """
        计算核函数
        :return: 高斯核矩阵
        """
        # 初始化高斯核结果矩阵 大小 = 训练集长度m * 训练集长度m
        # k[i][j] = Xi * Xj
        k = [[0 for i in range(self.m)] for j in range(self.m)]

        # 大循环遍历Xi，Xi为式7.90中的x
        for i in range(self.m):
            # 得到式7.90中的X
            X = self.train_data_mat[i, :]
            # 小循环遍历Xj，Xj为式7.90中的Z
            # 由于 Xi * Xj 等于 Xj * Xi，一次计算得到的结果可以
            # 同时放在k[i][j]和k[j][i]中，这样一个矩阵只需要计算一半即可
            # 所以小循环直接从i开始
            for j in range(i, self.m):
                # 获得Z
                Z = self.train_data_mat[j, :]
                # 先计算||X - Z||^2
                result = (X - Z) * (X - Z).T
                # 分子除以分母后去指数，得到的即为高斯核结果
                result = np.exp(-1 * result / (2 * self.sigma ** 2))
                # 由于是对称矩阵，所以将Xi*Xj的结果存放入k[i][j]和k[j][i]中
                k[i][j] = result
                k[j][i] = result
        # 返回高斯核矩阵
        return k

    def is_satisfy_KKT(self, i):
        """
        查看第i个α是否满足KKT条件
        :param i: α的下标
        :return: True/False
        """
        gxi = self.cal_gxi(i)
        yi = self.train_label_mat[i]

        # 判断依据参照“7.4.2 变量的选择方法”中“1.第1个变量的选择”
        # 依据7.111
        if (math.fabs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            return True
        # 依据7.113
        elif (math.fabs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True
        # 依据7.112
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) \
                and (math.fabs(yi * gxi - 1) < self.toler):
            return True

        return False

    def cal_gxi(self, i):
        """
        计算g(xi)
        根据“7.4.1 两个变量二次规划的求解方法” 式7.104
        :param i: x的下标
        :return: 
        """
        # 初始化g(xi)
        gxi = 0
        # 获得支持向量的index
        indexs = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        # 遍历每一个非零α，i为非零α的下标
        for index in indexs:
            # 计算g(xi)
            gxi += self.alpha[index] * self.train_label_mat[index] * self.k[index][i]
        # 求和结束后再加上偏置b
        gxi += self.b

        # 返回
        return gxi

    def cal_Ei(self, i):
        """
        计算Ei
        根据“7.4.1 两个变量二次规划的求解方法” 式7.105
        :param i: E的下标
        :return: 
        """
        # 计算g(xi)
        gxi = self.cal_gxi(i)
        # Ei = g(xi) - yi
        return gxi - self.train_label_mat[i]

    def get_alpha_j(self, E1, i):
        """
        SMO中选择第二个变量
        :param E1: 第一个变量的E1
        :param i: 第一个变量α的下标
        :return: E2，α2的下标
        """
        # 初始化E2
        E2 = 0
        # 初始化|E1-E2|为-1
        maxE1_E2 = -1
        # 初始化第二个变量的下标
        maxIndex = -1
        # 获得Ei非0的对应索引组成的列表，列表内容为非0Ei的下标i
        nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]
        # 对每个非零Ei的下标i进行遍历
        for j in nozeroE:
            # 计算E2
            E2_tmp = self.cal_Ei(j)
            # 如果|E1-E2|大于目前最大值
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                # 更新最大值
                maxE1_E2 = math.fabs(E1 - E2_tmp)
                # 更新最大值E2
                E2 = E2_tmp
                # 更新最大值E2的索引j
                maxIndex = j
        # 如果列表中没有非0元素了（对应程序最开始运行时的情况）
        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                # 获得随机数，如果随机数与第一个变量的下标i一致则重新随机
                maxIndex = int(random.uniform(0, self.m))
            # 获得E2
            E2 = self.cal_Ei(maxIndex)

        # 返回第二个变量的E2值以及其索引
        return E2, maxIndex

    def train(self, iter=100):
        """
        支持向量积训练
        :param iter: 迭代次数
        :return: 
        """
        # iter_step：迭代次数，超过设置次数还未收敛则提前中止
        iter_step = 0
        # parame_changed：单次迭代中有参数改变则增加1
        param_changed = 1

        # 如果没有达到迭代次数上限以及上次迭代中有参数改变则继续迭代
        # param_changed==0时表示上次迭代没有参数改变
        while (iter_step < iter) and (param_changed > 0):
            # 打印当前迭代轮数
            print('当前迭代轮数为：%d:%d' % (iter_step, iter))
            # 迭代步数加1
            iter_step += 1
            # 新的一轮将参数改变标志位重新置0
            param_changed = 0

            # 大循环遍历所有样本，用于找SMO中第一个变量
            for i in range(self.m):
                # 查看第一个遍历是否满足KKT条件，如果不满足则作为SMO中第一个变量从而进行优化
                if self.is_satisfy_KKT(i) == False:
                    # 第一个变量α的下标i已经确定，接下来按照“7.4.2 变量的选择方法”第二步选择变量2。
                    # 由于变量2的选择中涉及到|E1 - E2|，因此先计算E1
                    E1 = self.cal_Ei(i)

                    # 选择第2个变量
                    E2, j = self.get_alpha_j(E1, i)

                    # 参考“7.4.1两个变量二次规划的求解方法” P126 下半部分
                    # 获得两个变量的标签
                    y1 = self.train_label_mat[i]
                    y2 = self.train_label_mat[j]
                    # 复制α值作为old值
                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]
                    # 依据标签是否一致来生成不同的L和H
                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_2 + alphaOld_1)
                    # 如果两者相等，说明该变量无法再优化，直接跳到下一次循环
                    if L == H:
                        continue

                    # 计算α的新值
                    # 依据“7.4.1两个变量二次规划的求解方法”式7.106更新α2值
                    # 先获得几个k值，用来计算事7.106中的分母η
                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]
                    # 依据式7.106更新α2，该α2还未经剪切
                    alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)
                    # 剪切α2
                    if alphaNew_2 < L:
                        alphaNew_2 = L
                    elif alphaNew_2 > H:
                        alphaNew_2 = H
                    # 更新α1，依据式7.109
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)

                    # 依据“7.4.2 变量的选择方法”第三步式7.115和7.116计算b1和b2
                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k21 * (alphaNew_2 - alphaOld_2) + self.b
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k22 * (alphaNew_2 - alphaOld_2) + self.b

                    # 依据α1和α2的值范围确定新b
                    if (alphaNew_1 > 0) and (alphaNew_1 < self.C):
                        bNew = b1New
                    elif (alphaNew_2 > 0) and (alphaNew_2 < self.C):
                        bNew = b2New
                    else:
                        bNew = (b1New + b2New) / 2

                    # 将更新后的各值写入，进行更新
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew

                    self.E[i] = self.cal_Ei(i)
                    self.E[j] = self.cal_Ei(j)

                    # 如果α2的改变量过于小，就认为该参数未改变，不增加param_changed值
                    # 反之则自增1
                    if math.fabs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        param_changed += 1

                # 打印迭代轮数，i值，该迭代轮数，修改α数目
                print("当前迭代轮数为：%d i:%d, 修改α数目： %d" % (iter_step, i, param_changed))

        # 全部计算结束后，重新遍历一遍α，查找里面的支持向量
        for i in range(self.m):
            # 如果α>0，说明是支持向量
            if self.alpha[i] > 0:
                # 将支持向量的索引保存起来
                self.supportVecIndex.append(i)

    def cal_single_kernel(self, x1, x2):
        """
        单独计算核函数
        :param x1: 向量1
        :param x2: 向量2
        :return: 核函数结果
        """
        result = (x1 - x2) * (x1 - x2).T
        result = np.exp(-1 * result / (2 * self.sigma ** 2))
        # 返回结果
        return np.exp(result)

    def predict(self, x):
        """
        对样本进行预测
        :param x: 
        :return: 
        """
        result = 0
        for i in self.supportVecIndex:
            # 遍历所有支持向量，计算求和式
            # 如果是非支持向量，求和子式必为0，没有必须进行计算
            # 这也是为什么在SVM最后只有支持向量起作用
            # ------------------
            # 先单独将核函数计算出来
            tmp = self.cal_single_kernel(self.train_data_mat[i, :], np.mat(x))
            # 对每一项子式进行求和，最终计算得到求和项的值
            result += self.alpha[i] * self.train_label_mat[i] * tmp
        # 求和项计算结束后加上偏置b
        result += self.b
        # 使用sign函数(指示函数)返回预测结果
        return np.sign(result)

    def test(self, test_data, test_label):
        """
        测试
        :param test_data: 测试集
        :param test_label: 真实标签
        :return: 准确率
        """
        # 错误计数值
        error_count = 0
        # 遍历测试集所有样本
        for i in range(len(test_data)):
            # 打印目前进度
            print('test:%d:%d' % (i, len(test_data)))
            # 获取预测结果
            result = self.predict(test_data[i])
            # 如果预测与标签不一致，错误计数值加一
            if result != test_label[i]:
                error_count += 1
        # 返回正确率
        return 1 - error_count / len(test_data)


if __name__ == '__main__':
    start = time.time()

    # 获取训练集及标签
    print('starting to load data')
    train_data, train_label = load_data('../Input/mnist_train.csv')
    test_data, test_label = load_data('../Input/mnist_test.csv')

    # 初始化SVM类
    print('starting to init')
    svm = SVM(train_data[:1000], train_label[:1000], 10, 200, 0.001)

    # 开始训练
    print('starting to train')
    svm.train()

    # 开始测试
    print('starting to test')
    # 由于时间原因，仅选择100条测试数据测试
    accuracy = svm.test(test_data[:100], test_label[:100])
    print('准确率为：', accuracy)
    print('消耗时间为：', time.time() - start)

