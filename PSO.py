# coding: utf-8
import numpy as np
import random
import matplotlib.pyplot as plt

from map import distance_list

'''
粒子群优化算法

程序架构参考：https://zhuanlan.zhihu.com/p/482350842
代码思路参考：https://blog.csdn.net/baidu/article/details/124578774
'''


class PSO():
    '''
    城市序号0-99

    '''

    # PSO参数设置
    def __init__(self, pN, dim, max_iter):
        self.pN = pN  # 粒子数量，
        self.dim = dim  # 搜索维度,即结果的维度，应该与城市的数量相同
        self.w = 0.8  # 惯性权值
        # self.c1 = 2  # 学习因子，自我学习
        # self.c2 = 2  # 学习因子，群体最优学习
        self.r1 = 0.5  # 本地随机值
        self.r2 = 0.7  # 全局随机值
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim), dtype=int)  # 所有粒子的位置
        # self.V = np.zeros((self.pN, self.dim))  # 所有粒子的速度
        self.p_best = np.zeros((self.pN, self.dim), dtype=int)  # 个体经历的最佳位置
        self.g_best = np.zeros(self.dim, dtype=int)  # 全局最佳位置
        self.p_fit = np.zeros(self.pN, dtype=int)  # 每个个体的历史最佳适应值
        self.g_fit = 1000  # 全局最佳适应值

    def distance_sum(self, X_list):
        '''
        计算粒子的当前适应度，即该种方法目前的总距离
        :param X_list: 经过城市的顺序列表
        :return: 总距离
        '''
        distance = 0
        for index in range(len(X_list) - 1):
            distance += distance_list[X_list[index]][X_list[index + 1]]
        distance += distance_list[X_list[-1]][X_list[0]]
        return distance

    def get_ss(self, x_best, X_i, r):
        """
        计算交换序列ss，即x2结果交换序列ss得到x1，对应PSO速度更新公式中的 r1(p_best-xi) 和 r2(g_best-xi)
        :param x_best: p_best or g_best
        :param X_i: 粒子当前的解
        :param r: 对应的随机因子
        :return:
        """
        velocity_ss = []
        # r = r / (self.w + self.r1 + self.r2)
        for i in range(len(X_i)):
            if X_i[i] != x_best[i]:
                j = np.where(X_i == x_best[i])[0]
                so = (i, j, r)  # 得到交换子
                velocity_ss.append(so)
                X_i[i], X_i[j] = X_i[j], X_i[i]  # 执行交换操作

        return velocity_ss

    def do_ss(self, X_i, ss):
        """
        执行交换操作
        :param X_i:
        :param ss: 由交换子组成的交换序列
        :return:
        """
        for i, j, r in ss:
            rand = np.random.random()  # 产生一个0-1的浮点数
            if rand <= r:
                X_i[i], X_i[j] = X_i[j], X_i[i]
        return X_i

    # 初始化种群
    def init_Population(self):
        for i in range(self.pN):  # 因为要随机生成pN个数据，所以需要循环pN次
            self.X[i] = np.random.choice(list(range(self.dim)), size=self.dim, replace=False)
            self.p_best[i] = self.X[i]  # 其实就是给self.p_best定值，将当前值定为最优值
            tmp = self.distance_sum(self.X[i])  # 得到现在最优，tmp是当前适应值
            self.p_fit[i] = tmp  # 这个个体历史最佳的位置
            if tmp < self.g_fit:  # 得到现在最优和历史最优比较大小，如果现在最优大于历史最优，则更新历史最优
                self.g_fit = tmp
                self.g_best = self.X[i]

    # 更新粒子位置
    def iterator(self):
        fitness = []
        for step in range(self.max_iter):  # 迭代次数，不是越多越好
            for i in range(self.pN):  # 更新g_best和p_best
                temp = self.distance_sum(self.X[i])
                if temp < self.p_fit[i]:  # 更新个体最优
                    self.p_fit[i] = temp
                    self.p_best[i] = self.X[i]
                    if self.p_fit[i] < self.g_fit:  # 更新全局最优
                        self.g_best = self.X[i]
                        self.g_fit = self.p_fit[i]

                # 计算交换序列，即 v = r1(p_best-xi) + r2(g_best-xi)
                ss1 = self.get_ss(self.p_best[i], self.X[i], self.r1)
                ss2 = self.get_ss(self.g_best, self.X[i], self.r2)
                ss = ss1 + ss2
                self.X[i] = self.do_ss(self.X[i], ss)

            fitness.append(self.g_fit)
            # print(self.X[0], end=" ")
            print("第 {} 轮迭代，最优答案：".format(step), self.g_fit)  # 输出最优值
        return fitness


if __name__ == "__main__":
    # 程序
    max_iter = 150
    my_pso = PSO(pN=500, dim=len(distance_list), max_iter=max_iter)
    my_pso.init_Population()
    fitness = my_pso.iterator()
    # 输出结果
    print("路线：", my_pso.g_best)
    print("路径长度")
    for i in range(len(my_pso.g_best) - 1):
        print(distance_list[my_pso.g_best[i]][my_pso.g_best[i + 1]], " ", end="")
    print("\n总里程", my_pso.distance_sum(my_pso.g_best))
    # 画图
    plt.figure(1)
    plt.title("Figure1")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    t = np.array([t for t in range(0, max_iter)])
    fitness = np.array(fitness)
    plt.plot(t, fitness, color='b', linewidth=1)
    plt.show()
