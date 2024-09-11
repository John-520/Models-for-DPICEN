# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 19:12:18 2021

@author: 29792
"""
import numpy as np
import torch
import torch.nn as nn
from loss.DAN import DAN
from loss.MMD import mmd_rbf_noaccelerate, mmd_rbf_accelerate
from loss.JAN import JAN
from loss.MMD_loss import MMD_loss
from loss.CORAL import CORAL
import sys 
sys.path.append("D:\北京交通大学博士\论文【小】\论文【第四章】\code") 
from MMSD_main.MMSD import MMSD

from loss.lmmd import LMMD_loss
from loss.contrastive_center_loss import ContrastiveCenterLoss
from loss.SupervisedContrastiveLoss import SupervisedContrastiveLoss
from loss.ContrastiveLoss import ContrastiveLoss

from loss.SupConLoss import SupConLoss

import sub_models
import torch.nn.functional as F

from sklearn.cluster import KMeans

from loss.adv import *
from timm.loss import LabelSmoothingCrossEntropy

from loss.DANCE_loss import *
from sub_models.LinearAverage import LinearAverage

from torch.autograd import Variable
import matplotlib.pyplot as plt

import ot
from scipy.optimize import linear_sum_assignment



'行列正则化'
def l2row_torch(X):
	"""
	L2 normalize X by rows. We also use this to normalize by column with l2row(X.T)
	"""
	N = torch.sqrt((X**2).sum(axis=1)+1e-8)
	Y = (X.T/N).T
	return Y,N
 
BCEWithLogitsLoss = nn.BCEWithLogitsLoss()



class models(nn.Module):

    def __init__(self, args):
        super(models, self).__init__()

        self.args = args
        self.num_classes= args.num_classes
        self.bottle = nn.Sequential(nn.Linear(384, 150),  # 192
                                    nn.GELU(), 
                                    nn.Dropout()
                                    ) 

        self.cls_fc = nn.Linear(150, self.num_classes) #

        
        self.PEC_bottle = nn.Sequential(
                nn.Linear(384, 10),
            )
        
        self.ConvAutoencoderVAE_encoder = getattr(sub_models, 'ConvAutoencoderVAE_encoder')()
        self.ConvAutoencoderVAE_decoder = getattr(sub_models, 'ConvAutoencoderVAE_decoder')()
        
        self.ConvAutoencoder_encoder_PCE = getattr(sub_models, 'ConvAutoencoder_encoder_PCE')()
        self.ConvAutoencoder_decoder_PCE = getattr(sub_models, 'ConvAutoencoder_decoder_PCE')()
        
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
        


    def l2_zheng(self,x):
        x = torch.sqrt(x**2 + 1e-8)
        x = l2row_torch(x)[0]
        x = l2row_torch(x.T)[0].T        
        return x


    def forward(self, source,target,label_source,label_target, index_target, epoch,mu_value,task, ndata):
        loss = 0
        
        source_source = source[0]["hello"].cuda()
        source_PCE = source[3]["hello"].squeeze().cuda()
        
        s_pred_A = 0


        s_z = self.ConvAutoencoderVAE_encoder(source_source)   #  
        source_z_PCE = self.ConvAutoencoder_encoder_PCE(source_PCE) 
        source_z = s_z
        source_z = source_z.view(source_z.size(0), -1)
        source_PCE = source_PCE.view(source_PCE.size(0), -1)
        
        
        ########################################################融合
        f1= source_z + source_z_PCE
        f2= source_z + source_z_PCE
        features_f1 = f1
        features_f2 = f2
        f1 = f1.view(s_z.size())


        source_out = self.ConvAutoencoderVAE_decoder(f1) 
        loss_R1=F.mse_loss(source_source,source_out)
        
        
        source_out_PCE = self.ConvAutoencoder_decoder_PCE(f2)
        loss_R2=F.mse_loss(source_PCE,source_out_PCE)
        
        
        
        
        '均方误差'
        loss_PCE = F.mse_loss(source_z,source_z_PCE)

        
        

        '分类功能' ############################################################################################################
        
        source_z = self.l2_zheng(source_z)
        source = self.bottle(source_z)
        source = self.l2_zheng(source)
        s_pred = self.cls_fc(source)
        
        
        PCE_pred = self.PEC_bottle(source_z_PCE)
        loss_cls = F.nll_loss(F.log_softmax(PCE_pred, dim=1), label_source)

            
            
        if self.training == True:    
            
            if epoch< self.args.middle_epoch:
                loss_cls = 0
                loss_PCE = 0
                loss_PCEC = 0
                
            loss = loss_R1  + loss_R2 + loss_cls  + loss_PCE 
        



        if mu_value == 2:
            with open('results_miu.txt','a') as file0:
                print([task],mu_value,np.mean(loss_PCE.item()),file=file0)
        
        
        
        return source_out, source_source, s_pred, s_pred_A, _,   loss



    def predict(self, x):
        target_target = x[0]["hello"].cuda()
        x = self.ConvAutoencoderVAE_encoder(target_target) 
        x = x.view(x.size(0), -1)
        
        x = self.l2_zheng(x)
        x = self.bottle(x)


        x = self.l2_zheng(x)

        return self.cls_fc(x)
    



    
    