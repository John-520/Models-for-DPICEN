# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:37:11 2021

@author: 29792
"""
from functools import partial
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import torch.utils.data as data
import numpy as np
import torch
from torchsampler import ImbalancedDatasetSampler



#################################解决num_works != 0时的堵塞问题   没啥用
import cv2
cv2.setNumThreads(0)
#################################



class Mydataset(data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)
    def get_labels(self): 
        return self.y

def Make_data(X_train, Y_train, batch_size,shuffle=True):
    datasets = Mydataset(X_train, Y_train)  # 初始化

    dataloader = data.DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, num_workers=0) 
    return dataloader








class Mydataset_multi(data.Dataset):

    def __init__(self, x, y):

        self.x_0 = x[0]["hello"]
        self.x_1 = x[1]["hello"]
        self.x_2 = x[2]["hello"]
        self.x_3 = x[3]["hello"]
        self.x_4 = x[4]["hello"]
        self.x_5 = x[5]["hello"]
        
        self.y = y
        # self.idx = list()
        # for item in x:
        #     self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data_0 = self.x_0[index]
        input_data_1 = self.x_1[index]
        input_data_2 = self.x_2[index]
        input_data_3 = self.x_3[index]
        input_data_4 = self.x_4[index]
        input_data_5 = self.x_5[index]
        
        target = self.y[index]
        
        
        input_data = [
                        {"hello":input_data_0},
                        {"hello":input_data_1},
                        {"hello":input_data_2},
                        {"hello":input_data_3},
                        {"hello":input_data_4},
                        {"hello":input_data_5}
                      ]
        
        # input_data = [input_data_0,
        #               input_data_1,
        #               input_data_2,
        #               input_data_3,
        #               input_data_4,
        #               input_data_5
        #               ]
        
        
        return input_data, target, index
        # return input_data_0, input_data_1, input_data_2, input_data_3, input_data_4, input_data_5, target

    def __len__(self):
        return len(self.x_0)
    
    
    def get_labels(self): 
        return self.y

def Make_data_multi(X_train, Y_train, batch_size,shuffle=True):
    datasets = Mydataset_multi(X_train, Y_train)  # 初始化

    dataloader = data.DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, num_workers=0) 
    return dataloader
















# ########################测试字典数据是否可以直接输入到dataloader中。结论是可以的哈哈
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
# A = {"hello":np.ones((10))}
# B = {"hello":np.ones((10))*2}
# listP =[A,B]
# class Adata(Dataset):
#     def __init__(self):
#         super().__init__()
#         self.listP = listP
#     def __getitem__(self, idx):
#         return listP[idx]
#     def __len__(self):
#         return len(self.listP[0]["hello"])
# P = Adata()

# train_dl = DataLoader(P,batch_size=2)



# # X = train_dl["hello"]


# ci = 0
# for i in train_dl:
#     ci = ci+1 
#     # aa = i["hello"][0]  # [0:1500]
#     # bb = i[1500:3000]
#     print(ci,i)
#     # print(i)
# # ['hello': tensor([1，2])]







# import torch
# from torch.utils.data import Dataset

# class MyDataset(Dataset):
#     # 构造函数
#     def __init__(self, data_tensor, target_tensor):
#         self.data_tensor = data_tensor
#         self.target_tensor = target_tensor
#     # 返回数据集大小
#     def __len__(self):
#         return self.data_tensor.size(0)
#     # 返回索引的数据与标签
#     def __getitem__(self, index):
#         return self.data_tensor[index], self.target_tensor[index]





# =============================================================================
# 
# class Mydataset(data.Dataset):
# 
#     def __init__(self, x):
#         self.x = x
#         self.idx = list()
#         for item in x:
#             self.idx.append(item)
#         pass
# 
#     def __getitem__(self, index):
#         input_data = self.idx[index]
#         return input_data
# 
#     def __len__(self):
#         return len(self.idx)
# 
# def Make_data(input_data,batch_size):
#     datasets = Mydataset(input_data)  # 初始化
# 
#     dataloader = data.DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=0) 
#     return dataloader
#     
# =============================================================================
    
# =============================================================================
# if __name__ ==('__main__'):
# 
#     datasets = Mydataset(X_train, Y_train)  # 初始化
# 
#     dataloader = data.DataLoader(datasets, batch_size=100, shuffle=True, num_workers=0) 
# 
#     for i, (input_data, target) in enumerate(dataloader):
# # =============================================================================
# #         print('input_data%d' % i, input_data)
# #         print('target%d' % i, target)
# # =============================================================================
#         print(input_data.shape,target.shape)
# =============================================================================



