# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 7:50 PM
# @Author  : WittonZhou
# @File    : HMM.py

import numpy as np
import time
'''
HMM模型
'''


def load_data(filename):
    """
    加载数据集
    :param filename: 
    :return: 
    """
    artical = []
    with open(filename,  encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            artical.append(line)
    return artical


def train(filename):
    """
    依据训练文本统计PI、A、B
    :param filename: 
    :return: 
    """
    # 定义一个查询字典，用于映射四种标记在数组中对应的位置，方便查询
    # B：词语的开头
    # M：一个词语的中间词
    # E：一个词语的结果
    # S：非词语，单个词
    statusDict = {'B': 0, 'M': 1, 'E': 2, 'S': 3}

    # 每个字只有四种状态，所以下方的各类初始化中大小的参数均为4
    # 初始化PI的一维数组，因为对应四种状态，大小为4
    PI = np.zeros(4)
    # 初始化状态转移矩阵A，涉及到四种状态各自到四种状态的转移，因为大小为4x4
    A = np.zeros((4, 4))
    # 初始化观测概率矩阵，分别为四种状态到每个字的发射概率
    # 因为是中文分词，使用ord(汉字)即可找到其对应编码，这里用一个65536的空间来保证对于所有的汉字都能
    # 找到对应的位置来存储
    B = np.zeros((4, 65536))
    # 读训练文本
    with open(filename, encoding='utf-8') as f:
        # 文本中的每一行认为是一个训练样本
        # 在统计上，三个参数依据“10.3.2” Baum-Welch算法内描述的统计
        # PI依据式10.35
        # A依据10.37
        # B依据10.38
        # 注：并没有使用Baum-Welch算法，只是借助了其内部的三个参数生成公式，其实
        # 公式并不是Baum-Welch特有的，只是在那一节正好有描述
        for line in f.readlines():
            # 对单行句子按空格进行切割
            current_line = line.strip().split()
            # 对词性的标记放在该列表中
            wordLabel = []
            # 对每一个单词进行遍历
            for i in range(len(current_line)):
                # 如果长度为1，则直接将该字标记为S，即单个词
                if len(current_line[i]) == 1:
                    label = 'S'
                else:
                    # 如果长度不为1，开头为B，最后为E，中间添加长度-2个M
                    # 如果长度刚好为2，长度-2=0也就不添加了，反之添加对应个数的M
                    label = 'B' + 'M' * (len(current_line[i]) - 2) + 'E'

                # 如果是单行开头第一个字，PI中对应位置加1,
                if i == 0:
                    PI[statusDict[label[0]]] += 1

                # 对于该单词中的每一个字，在生成的状态链中统计B
                for j in range(len(label)):
                    # 遍历状态链中每一个状态，并找到对应的中文汉字，在B中对应位置加1
                    B[statusDict[label[j]]][ord(current_line[i][j])] += 1

                # 在整行的状态链中添加该单词的状态链
                # 注意：extend表直接在原先元素的后方添加，可以百度一下extend和append的区别
                wordLabel.extend(label)

            # 单行所有单词都结束后，统计A信息
            # 因为A涉及到前一个状态，因此需要等整条状态链都生成了才能开始统计
            for i in range(1, len(wordLabel)):
                # 统计t时刻状态和t-1时刻状态的所有状态组合的出现次数
                A[statusDict[wordLabel[i - 1]]][statusDict[wordLabel[i]]] += 1

                # 上面代码在统计上全部是统计的次数，实际运算需要使用概率，
                # 下方代码是将三个参数的次数转换为概率
                # ----------------------------------------
                # 对PI求和，概率生成中的分母
    sum = np.sum(PI)
    # 遍历PI中每一个元素，元素出现的次数/总次数即为概率
    for i in range(len(PI)):
        # 如果某元素没有出现过，该位置为0，在后续的计算中这是不被允许的
        # 比如说某个汉字在训练集中没有出现过，那在后续不同概率相乘中只要有
        # 一项为0，其他都是0了，此外整条链很长的情况下，太多0-1的概率相乘
        # 不管怎样最后的结果都会很小，很容易下溢出
        # 所以在概率上我们习惯将其转换为log对数形式，这在书上是没有讲的
        # x大的时候，log也大，x小的时候，log也相应小，我们最后比较的是不同
        # 概率的大小，所以使用log没有问题

        # 那么当单向概率为0的时候，log没有定义，因此需要单独判断
        # 如果该项为0，则手动赋予一个极小值
        if PI[i] == 0:
            PI[i] = -3.14e+100
        # 如果不为0，则计算概率，再对概率求log
        else:
            PI[i] = np.log(PI[i] / sum)

    # 与上方PI思路一样，求得A的概率对数
    for i in range(len(A)):
        sum = np.sum(A[i])
        for j in range(len(A[i])):
            if A[i][j] == 0:
                A[i][j] = -3.14e+100
            else:
                A[i][j] = np.log(A[i][j] / sum)

    # 与上方PI思路一样，求得B的概率对数
    for i in range(len(B)):
        sum = np.sum(len(B[i]))
        for j in range(len(B[i])):
            if B[i][j] == 0:
                B[i][j] = -3.14e+100
            else:
                B[i][j] = np.log(B[i][j] / sum)

    # 返回统计得到的三个参数
    return PI, A, B


def participle(artical, PI, A, B):
    """
    分词
    算法依据“10.4.2 维特比算法”
    :param artical: 要分词的文章
    :param PI: 初始状态概率向量PI
    :param A: 状态转移矩阵
    :param B: 观测概率矩阵
    :return: 分词后的文章
    """
    part_artical = []
    for line in artical:
        # 初始化δ，δ存放四种状态的概率值，因为状态链中每个状态都有
        delta = [[0 for i in range(4)] for i in range(len(line))]
        for i in range(4):
            # 初始化δ状态链中第一个状态的四种状态概率
            delta[0][i] = PI[i] + B[i][ord(line[0])]
        # 初始化ψ，初始时为0
        psi = [[0 for i in range(4)] for i in range(len(line))]

        for t in range(1, len(line)):
            for i in range(4):
                tmpDelta = [0] * 4
                for j in range(4):
                    tmpDelta[j] = delta[t - 1][j] + A[j][i]
                # 找到最大的那个δ * a，
                maxDelta = max(tmpDelta)
                # 记录最大值对应的状态
                maxDeltaIndex = tmpDelta.index(maxDelta)

                # 将找到的最大值乘以b放入，
                # 注意：这里同样因为log变成了加法
                delta[t][i] = maxDelta + B[i][ord(line[t])]
                # 在ψ中记录对应的最大状态索引
                psi[t][i] = maxDeltaIndex

        # 建立一个状态链列表，开始生成状态链
        sequence = []
        # 算法10.5 第三步：终止
        # 在上面for循环全部结束后，很明显就到了第三步了
        # 获取最后一个状态的最大状态概率对应的索引
        i_opt = delta[len(line) - 1].index(max(delta[len(line) - 1]))
        # 在状态链中添加索引
        # 注：状态链应该是B、M、E、S，这里图方便用了0、1、2、3，其实一样的
        sequence.append(i_opt)
        # 算法10.5 第四步：最优路径回溯
        # 从后往前遍历整条链
        for t in range(len(line) - 1, 0, -1):
            # 不断地从当前时刻t的ψ列表中读取到t-1的最优状态
            i_opt = psi[t][i_opt]
            # 将状态放入列表中
            sequence.append(i_opt)
        # 因为是从后往前将状态放入的列表，所以这里需要翻转一下，变成了从前往后
        sequence.reverse()

        # 开始对该行分词
        current_line = ''
        # 遍历该行每一个字
        for i in range(len(line)):
            # 在列表中放入该字
            current_line += line[i]
            # 如果该字是3：S->单个词  或  2:E->结尾词 ，则在该字后面加上分隔符 |
            # 此外如果改行的最后一个字了，也就不需要加 |
            if (sequence[i] == 3 or sequence[i] == 2) and i != (len(line) - 1):
                current_line += '|'
        # 在返回列表中添加分词后的该行
        part_artical.append(current_line)
    # 返回分词后的文章
    return part_artical


if __name__ == '__main__':
    start = time.time()
    artical = load_data('test.txt')
    PI, A, B = train('train.txt')

    # 打印原文
    print('原文如下：')
    for line in artical:
        print(line)

    # 进行分词
    part_artical = participle(artical, PI, A, B)

    # 打印结果
    print('分词后：')
    for line in part_artical:
        print(line)

    print('消耗时间为：', time.time() - start)


