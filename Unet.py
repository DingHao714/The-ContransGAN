import torch
import torch.nn as nn
from attationmap import MultiHeadSelfAttention
from torchsummary import summary
# class BasicBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_ch)
#         self.relu= nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_ch)
#
#      
#         if out_ch != in_ch:
#             self.conv1x1 = nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=1)
#         else :
#             self.conv1x1=None
#     def forward(self, x):
#         out1 = self.relu((self.bn1(self.conv1(x))))
#         out = self.bn2(self.conv2(out1))
#         if self.conv1x1:
#             x = self.conv1x1(x)
#
#         out = self.relu(out + x)
#         return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_ch, out_ch, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_ch)
#         self.ru=nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_ch)
#
#         self.extra = nn.Sequential()
#         
#         if out_ch != in_ch:
#             self.extra = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
#                 nn.BatchNorm2d(out_ch)
#             )
#
#     def forward(self, x):
#         out = self.ru((self.bn1(self.conv1(x))))
#         out = self.bn2(self.conv2(out))
#         out = self.extra(x) + out
#         return out

class double_conv(nn.Module):
            '''(conv => BN => ReLU) * 2'''
            def __init__(self, in_ch, out_ch):
                super(double_conv, self).__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)

                )
            def forward(self, input):
                input = self.conv(input)
                return input


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)  # 输入通道数in_ch为3， 输出通道数out_ch为64

    def forward(self, input):
        input = self.conv(input)
        return input



class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, input):
        input = self.mpconv(input)
        return input


class up(nn.Module):
    def __init__(self, in_ch,mid_ch, out_ch):
        super(up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch, mid_ch, 2, stride=2)

        self.conv = double_conv(2*mid_ch, out_ch)

    def forward(self, x1, x2): 
       
        x1 = self.up(x1)

        input = torch.cat([x2, x1], dim=1)
        input = self.conv(input)
        return input

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, input):
        input = self.conv(input)
        return input



class unetno(nn.Module):
    def __init__(self, in_ch, out_ch):  
            super(unetno, self).__init__()
            self.inc = inconv(in_ch,96)
            self.down1 = down(96, 192)
            self.down2 = down(192, 384)
            self.down3 = down(384, 768)
            self.down4 = down(768, 1536)
            self.up1 = up(1536, 768, 768)
            self.up2 = up(768,384, 384)
            self.up3 = up(384,192, 192)
            self.up4 = up(192,96, 96)
            self.outc = outconv(96,out_ch)

    def forward(self, x):
            x1 = self.inc(x)

            x2 = self.down1(x1)

            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x = self.outc(x)
            return x
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model = SwinTransformer(hidden_dim=96, layers=(2,2,2,2), heads=(3,6,12,24)).to(device)
# model = unetno(1,1).to(device)
#
# summary(model, (1, 224, 2048))
