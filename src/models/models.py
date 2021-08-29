import torch
import torch.nn as nn 
import torch.nn.functional as F

# https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html
class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

class SDC_Block(nn.Module):

    def __init__(self, in_c, out_c):

        super(SDC_Block,self).__init__()

        self.conv1 = nn.Conv2d(in_c,out_c, kernel_size=15, stride=1, padding=7, bias=True, dilation=1)
        self.conv2 = nn.Conv2d(in_c,out_c, kernel_size=15, stride=1, padding=14, bias=True, dilation=2)
        self.conv3 = nn.Conv2d(in_c,out_c, kernel_size=15, stride=1, padding=21, bias=True, dilation=3)

        self.bn = nn.BatchNorm2d(out_c * 3)
        self.relu = nn.ReLU()

    def forward(self,x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        result = torch.cat((x1,x2,x3),dim=1)

        result = self.relu(self.bn(result))

        return result


class Basic_Block(nn.Module):

    def __init__(self, c_in, c_out):
        super(Basic_Block,self).__init__()

        self.non_linearity = nn.ReLU()

        self.c = nn.Conv2d(c_in, c_out, kernel_size=7, padding=3)
        self.bc = nn.BatchNorm2d(c_out)

        self.r = nn.Conv2d(c_out, c_in, kernel_size=1)
        self.br = nn.BatchNorm2d(c_in)
    
    def forward(self,x):
        identity = x
        x = self.non_linearity(self.bc(self.c(x)))
        x = self.non_linearity(self.br(self.r(x)))
        x = x + identity
        return x


    

class CTM(nn.Module):

    def __init__(self, cfg):

        super(CTM,self).__init__()

        self.nonlinearity = nn.ReLU() ##nn.Sigmoid() #nn.ReLU()

        self.mult = 3
        self.num_channels = len(cfg.channels)
        self.intermediate_layers = 4

        self.iln = nn.LayerNorm((self.num_channels,512,512))

        layers = []
        for i in range(0,self.intermediate_layers):
            layers.append(Basic_Block(self.num_channels, self.num_channels * self.mult))

        self.layers = nn.Sequential(*layers)
        
        self.cls = nn.Conv2d(self.num_channels, 1, kernel_size=3, padding=1)

    def forward(self,x):

        x = self.iln(x)

        x = self.layers(x)

        x = self.cls(x)

        return x


