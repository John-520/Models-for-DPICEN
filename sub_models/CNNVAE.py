# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 10:48:58 2023

@author: luzy1
"""
import torch
from torch import nn
import torch.nn.functional as F
class ConvAutoencoder_encoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_encoder, self).__init__()

        #编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=81, stride=8, padding=1, dilation=1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )


        #潜在空间的维度
        self.latent_dim = 384
        #均值和方差的全连接层
        self.fc_mu = nn.Linear(384, self.latent_dim)
        self.fc_logvar = nn.Linear(384, self.latent_dim)


    
    def forward(self, x):
        x = self.encoder(x)
        s_z = x

        return s_z 
    
    

    
    
    
    

class ConvAutoencoder_decoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_decoder, self).__init__()


        #解码器       
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            nn.ConvTranspose1d(16, 1, kernel_size=80, stride=8, padding=0, output_padding=0),
        )




    def forward(self, x):
        x = self.decoder(x)
        return x
    
    
    
    
    


    
