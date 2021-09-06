import torch
import torch.nn as nn 
import torch.nn.functional as F

import segmentation_models_pytorch as smp

import torchvision
import torch
from torch import Tensor
import torch.nn as nn

from typing import Type, Any, Callable, Union, List, Optional


class BasicBlock(nn.Module):

    def __init__(self, in_c):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size = 1, padding = 0)
        self.bn1 = norm_layer(in_c)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_c, in_c, kernel_size = 3, padding = 1)
        self.bn2 = norm_layer(in_c)


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out.clone() + identity
        out = self.relu(out)

        return out

class SDCBlock(nn.Module):

    def __init__(self, in_c):
        super(SDCBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        
        self.conv1 = nn.Conv2d(in_c, in_c, dilation = 1, kernel_size = 5, padding = 2)
        self.conv2 = nn.Conv2d(in_c, in_c, dilation = 2, kernel_size = 5, padding = 4)
        self.conv3 = nn.Conv2d(in_c, in_c, dilation = 3, kernel_size = 5, padding = 6)

        self.conv = nn.Conv2d(in_c * 3, in_c, kernel_size = 1)

        self.bn1 = norm_layer(in_c * 3)
        self.bn2 = norm_layer(in_c)

        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)

        out = torch.cat((out1,out2,out3),dim=1)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = out.clone() + identity
        out = self.relu(out)

        return out


class Net(nn.Module):

    def __init__(self, cfg):
        super(Net, self).__init__()
        
        #norm_layer = nn.BatchNorm2d
        #self._norm_layer = norm_layer

        self.in_c = cfg.channels
        self.inter_c = 9
        
        #self.sp = nn.Dropout2d(p=0.2, inplace=False)
        #self.ln = nn.LayerNorm((512,512))

        #self.layer_freq = self._make_layer(BasicBlock, 6, 6)
        #self.layer_elevation = self._make_layer(BasicBlock, 1, 6)

        #self.layer1 = self._make_layer(BasicBlock, self.inter_c, 3)
        #self.layer2 = self._make_layer(SDCBlock, self.inter_c, 3)
        
        
        #self.fc = nn.Conv2d(self.inter_c, 1, kernel_size = 1)
        '''

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if True:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0) 
        ''' 
        self.layer_freq = getattr(smp,'Unet')(
            encoder_name='resnet18',       
            encoder_weights="imagenet", 
            encoder_depth=3,
            decoder_channels=(64, 32, 16),
            activation=None,
            decoder_attention_type=None,
            decoder_use_batchnorm=True,
            in_channels=6,                  
            classes=2,
        )

        self.layer_elevation = getattr(smp,'Unet')(
            encoder_name='resnet18',       
            encoder_weights="imagenet", 
            encoder_depth=3,
            decoder_channels=(64, 32, 16),
            activation=None,
            decoder_attention_type=None,
            decoder_use_batchnorm=True,
            in_channels=1,                  
            classes=2,
        )

        self.inter_model = getattr(smp,'Unet')(
            encoder_name='resnet34',       
            encoder_weights="imagenet", 
            encoder_depth=5,
            decoder_channels=(256,128,64, 32, 16),
            activation=None,
            decoder_attention_type=None,
            decoder_use_batchnorm=True,
            in_channels=6,                  
            classes=1,
        )

    def _make_layer(self, block, in_c, blocks):
        norm_layer = self._norm_layer
        
        layers = []
        for _ in range(0, blocks):
            layers.append(block(in_c))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        
        identity = x

        x_vvh = x[:,:2].clone()
        x_freq = x[:,2:-1].clone()
        x_elevation = x[:,-1].clone().unsqueeze(1)

        x_freq = self.layer_freq(x_freq)
        x_elevation = self.layer_elevation(x_elevation)

        x = torch.cat((x_vvh, x_freq, x_elevation), dim=1)

        x = self.inter_model(x)

        #x = self.sp(x)
        
        #x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)
        #x = self.layer5(x)
        #x = self.layer6(x)
        
        #x = self.fc(x)

        return x

    def forward(self, x):
        #with torch.autograd.set_detect_anomaly(True):
        return self._forward_impl(x)
