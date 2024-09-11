# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 17:24:13 2022

@author: 29792
"""

''' ============== 特征提取的类 =====================

参考文献 英文文献 016_C_(Q1 时域和频域共24种特征参数 )  Fault diagnosis of rotating machinery based on multiple ANFIS combination with GAs

'''
import math
import numpy as np
import scipy.stats

class Fea_Extra():
    def __init__(self, Signal, Fs = 25600):
        self.signal = Signal
        self.Fs = Fs

    def Fre_fea(self, signal_):
        """
        提取频域特征 13类
        :param signal_:
        :return:
        """
        L = len(signal_)
        PL = abs(np.fft.fft(signal_ / L))[: int(L / 2)]
        PL[0] = 0
        f = np.fft.fftfreq(L, 1 / self.Fs)[: int(L / 2)]
        x = f
        y = PL
        K = len(y)


        f_12 = np.mean(y)

        f_13 = np.var(y)

        f_14 = (np.sum((y - f_12)**3))/(K * ((np.sqrt(f_13))**3))

        f_15 = (np.sum((y - f_12)**4))/(K * ((f_13)**2))

        f_16 = (np.sum(x * y))/(np.sum(y))

        f_17 = np.sqrt((np.mean(((x- f_16)**2)*(y))))

        f_18 = np.sqrt((np.sum((x**2)*y))/(np.sum(y)))

        f_19 = np.sqrt((np.sum((x**4)*y))/(np.sum((x**2)*y)))

        f_20 = (np.sum((x**2)*y))/(np.sqrt((np.sum(y))*(np.sum((x**4)*y))))

        f_21 = f_17/f_16

        f_22 = (np.sum(((x - f_16)**3)*y))/(K * (f_17**3))

        f_23 = (np.sum(((x - f_16)**4)*y))/(K * (f_17**4))
        f_fea = np.array([f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23])

        return f_fea

    def Both_Fea(self):
        """
        :return:  array
        """
        f_fea = self.Fre_fea(self.signal)
        tf_fea = get_wavelet_packet_feature(self.signal)
        fea = np.concatenate((f_fea,  tf_fea))

        return fea




import pywt
import numpy as np


def get_wavelet_packet_feature(data, wavelet='db2', mode='symmetric', maxlevel=3):  # db3
    """
    提取 小波包特征
    
    @param data: shape 为 (n, ) 的 1D array 数据，其中，n 为样本（信号）长度
    @return: 最后一层 子频带 的 能量百分比
    """
    wp = pywt.WaveletPacket(data, wavelet=wavelet, mode=mode, maxlevel=maxlevel)
    
    nodes = [node.path for node in wp.get_level(maxlevel, 'natural')]  # 获得最后一层的节点路径
    
    e_i_list = []  # 节点能量
    for node in nodes:
        e_i = np.linalg.norm(wp[node].data, ord=None) ** 2  # 求 2范数，再开平方，得到 频段的能量（能量=信号的平方和）
        e_i_list.append(e_i)
    
    # 以 频段 能量 作为特征向量
    # features = e_i_list
        
    # 以 能量百分比 作为特征向量，能量值有时算出来会比较大，因而通过计算能量百分比将其进行缩放至 0~100 之间
    e_total = np.sum(e_i_list)  # 总能量
    features = []
    for e_i in e_i_list:
        features.append(e_i / e_total * 100)  # 能量百分比
    
    return np.array(features)



