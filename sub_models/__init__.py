#!/usr/bin/python
# -*- coding:utf-8 -*-
from sub_models.cnn_1d import cnn_features as cnn_features_1d
from sub_models.MIXCNN import MIXCNN as MIXCNN

from sub_models.MLP import MLP as MLP
from sub_models.CNN1d import CNN1d as CNN1d
from sub_models.AdversarialNet import AdversarialNet
from sub_models.resnet18_1d import resnet18_features as resnet_features_1d
from sub_models.Resnet1d import resnet18 as resnet_1d

from sub_models.CNNAE import ConvAutoencoder_encoder as ConvAutoencoder_encoder
from sub_models.CNNAE import ConvAutoencoder_decoder as ConvAutoencoder_decoder

from sub_models.CNNVAE import ConvAutoencoder_encoder as ConvAutoencoderVAE_encoder
from sub_models.CNNVAE import ConvAutoencoder_decoder as ConvAutoencoderVAE_decoder

from sub_models.CNNAE_PCE import ConvAutoencoder_encoder as ConvAutoencoder_encoder_PCE
from sub_models.CNNAE_PCE import ConvAutoencoder_decoder as ConvAutoencoder_decoder_PCE