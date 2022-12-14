import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=2,output_norm=1,abstract_pool_size=1):
        super(U_Net,self).__init__()
        self.output_norm = output_norm
        self.abstract_pool_size = abstract_pool_size

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=512)
        self.Conv6 = conv_block(ch_in=512,ch_out=512)
        self.Conv7 = conv_block(ch_in=512,ch_out=512)
        self.Conv8 = conv_block(ch_in=512,ch_out=512)

        self.Up8 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv8 = conv_block(ch_in=768, ch_out=512)

        self.Up7 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv7 = conv_block(ch_in=768,ch_out=512)

        self.Up6 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv6 = conv_block(ch_in=768,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv5 = conv_block(ch_in=768, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def print_grad(self):
        print(self.Conv1)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        x6 = self.Maxpool(x5)
        x6 = self.Conv6(x6)

        x7 = self.Maxpool(x6)
        x7 = self.Conv7(x7)

        x8 = self.Maxpool(x7)
        x8 = self.Conv8(x8)


        # decoding + concat path
        out = self.Up8(x8)
        out = torch.cat((x7,out),dim=1)
        out = self.Up_conv8(out)

        out = self.Up7(out)
        out = torch.cat((x6,out),dim=1)
        out = self.Up_conv7(out)

        out = self.Up6(out)
        out = torch.cat((x5,out),dim=1)
        out = self.Up_conv6(out)

        out = self.Up5(out)
        out = torch.cat((x4,out),dim=1)
        out = self.Up_conv5(out)

        out = self.Up4(out)
        out = torch.cat((x3,out),dim=1)
        out = self.Up_conv4(out)

        out = self.Up3(out)
        out = torch.cat((x2,out),dim=1)
        out = self.Up_conv3(out)

        out = self.Up2(out)
        out = torch.cat((x1,out),dim=1)
        out = self.Up_conv2(out)

        out = self.Conv_1x1(out)
        out = out / self.output_norm

        return out
