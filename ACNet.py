import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from convmixer import *
from do_conv_pytorch import DOConv2d

torch.manual_seed(970530)  # 设置 (CPU) 生成随机数的种子，并返回一个torch.Generator对象。生成的随机数是固定的。
torch.cuda.manual_seed_all(970530)  # 设置 (GPU) 生成随机数的种子，并返回一个torch.Generator对象。

# scSE
# sSE 空间注意力
class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
    def forward(self, U):
        q = self.Conv1x1(U) # U:[bs,c,h,w] to q:[bs,1,h,w]
        spaceAtten = q  # 空间注意力
        spaceAtten = torch.squeeze(spaceAtten, 1)  # spaceAtten:[bs,h,w]
        q = self.norm(q)
        return U * q  # 广播机制 返回空间注意力

# cSE 通道注意力
class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2,
        kernel_size = 1, bias = False)  # in_channels // 取整运算
        self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels,
        kernel_size = 1, bias = False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)  # shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z)  # shape: [bs, c//2,1,1] # 通道注意力
        z = self.Conv_Excitation(z)  # shape: [bs, c,1,1]
        channelAtten = z  # 通道注意力
        channelAtten = torch.squeeze(channelAtten, 3)  # shape: [bs,c,1]
        z = self.norm(z)
        return U * z.expand_as(U)  # channelAtten 通道注意力;z.expand_as(U)扩展到U的维度

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        ### U_cse + U_sse:[]
        return U_cse + U_sse # 返回空间注意力和通道注意力


# h-swish激活函数
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


# def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
#     """
#     Add a conv2d / batchnorm / leaky ReLU block.
#     Args:
#         in_ch (int): number of input channels of the convolution layer.
#         out_ch (int): number of output channels of the convolution layer.
#         ksize (int): kernel size of the convolution layer.
#         stride (int): stride of the convolution layer.
#     Returns:
#         stage (Sequential) : Sequential layers composing a convolution block.
#     """
#     stage = nn.Sequential()
#     pad = (ksize - 1) // 2
#     stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
#                                        out_channels=out_ch, kernel_size=ksize, stride=stride,
#                                        padding=pad, bias=False))
#     stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
#     if leaky:
#         stage.add_module('leaky', nn.LeakyReLU(0.1))
#     else:
#         stage.add_module('relu6', nn.ReLU6(inplace=True))
#     return stage
#
#
# class ASFF(nn.Module):
#     def __init__(self, level, rfb=False, vis=False):
#         super(ASFF, self).__init__()
#         self.level = level
#         self.dim = [64, 64, 64]
#         self.inter_dim = self.dim[self.level]
#         # 每个level融合前，需要先调整到一样的尺度
#         if level == 0:
#             self.stride_level_1 = torch.nn.Conv2d(128, self.inter_dim, 3, 2)
#             self.stride_level_2 = torch.nn.Conv2d(128, self.inter_dim, 3, 2)
#             self.expand = torch.nn.Conv2d(self.inter_dim, 128, 3, 1)
#         elif level == 1:
#             self.compress_level_0 = torch.nn.Conv2d(128, self.inter_dim, 1, 1)
#             self.stride_level_2 = torch.nn.Conv2d(128, self.inter_dim, 3, 2)
#             self.expand = torch.nn.Conv2d(self.inter_dim, 128, 3, 1)
#
#         elif level == 2:
#             self.compress_level_0 = torch.nn.Conv2d(128, self.inter_dim, 1, 1)
#             self.expand = torch.nn.Conv2d(self.inter_dim, 128, 3, 1)
#
#
#         compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory
#
#         self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
#
#         self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
#         self.vis = vis
#
#
#     def forward(self, x_level_0, x_level_1, x_level_2):
#         if self.level == 0:
#             level_0_resized = x_level_0
#             level_1_resized = self.stride_level_1(x_level_1)
#
#             level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
#             level_2_resized = self.stride_level_2(level_2_downsampled_inter)
#
#         elif self.level == 1:
#             level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
#             level_1_resized = x_level_1
#             level_2_resized = self.stride_level_2(x_level_2)
#         elif self.level == 2:
#             level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
#             level_1_resized = F.interpolate(x_level_1, scale_factor=2, mode='nearest')
#             level_2_resized = x_level_2
#
#         level_0_weight_v = self.weight_level_0(level_0_resized)
#         level_1_weight_v = self.weight_level_1(level_1_resized)
#         level_2_weight_v = self.weight_level_2(level_2_resized)
#         levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
#         # 学习的3个尺度权重
#         levels_weight = self.weight_levels(levels_weight_v)
#         levels_weight = F.softmax(levels_weight, dim=1)
#         # 自适应权重融合
#         fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
#                             level_1_resized * levels_weight[:, 1:2, :, :] + \
#                             level_2_resized * levels_weight[:, 2:, :, :]
#
#         out = self.expand(fused_out_reduced)
#
#         if self.vis:
#             return out, levels_weight, fused_out_reduced.sum(dim=1)
#         else:
#             return out

class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        """
        :params c1: 输入feature map的通道数
        :params c2: 输出的feature map的通道数（这个函数的关键点就是对c2进行分组）
        :params k: 混合的卷积核大小 其实一般是[3, 5, 7...]用的比较多的
        :params s: 步长 stride
        :params equal_ch: 通道划分方式 有均等划分和指数划分两种方式  默认是均等划分
        """
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # 均等划分通道
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # 指数划分通道
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # 这里核原论文略有出入，这里加了一个shortcut操作
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))

class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv3x3BNReLU, self).__init__()
        # self.conv3x3 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.conv3x3 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)  # inplace = True ,会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
    def forward(self, x):
        return self.relu(self.bn(self.conv3x3(x)))

class SSHContextModule(nn.Module):
    def __init__(self, in_channel):
        super(SSHContextModule, self).__init__()
        self.stem = Conv3x3BNReLU(in_channel, in_channel//2)
        self.branch1_conv3x3 = Conv3x3BNReLU(in_channel//2, in_channel//2)
        self.branch2_conv3x3_1 = Conv3x3BNReLU(in_channel//2, in_channel//2)
        self.branch2_conv3x3_2 = Conv3x3BNReLU(in_channel//2, in_channel//2)
    def forward(self, x):
        x = self.stem(x)
        # branch1
        x1 = self.branch1_conv3x3(x)
        # branch2
        x2 = self.branch2_conv3x3_1(x)
        x2 = self.branch2_conv3x3_2(x2)
        # concat
        # print(x1.shape, x2.shape)
        return torch.cat([x1, x2], dim=1)


class EARCNN(nn.Module):
    def __init__(self, num_classes):
        super(EARCNN, self).__init__()
        self.Atten = scSE(in_channels=8)
        self.bneck = nn.Sequential(     #  初始 x = [50670, 6, 8, 9, 5]  分离后这里输入实际 x1 = [50670, 8, 9, 5]
            torch.nn.Conv2d(8, 32, kernel_size=3, padding=1),  # x = [50670, 32, 9, 5]

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            DOConv2d(64, 64, kernel_size=3, stride=2, padding=1),  # x = [50670, 64, 9, 5]
            nn.ReLU(),

            nn.Dropout2d(0.3),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),  # x = [50670, 128, 9, 5]
            DOConv2d(128, 128, kernel_size=3, stride=2, padding=1),  # x = [50670, 64, 9, 5]
            nn.ReLU(),

            # SSH模块 ↓
            SSHContextModule(128),  # x = [50670, 128, 9, 5]
            # SSH模块 ↑

            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),  # x = [50670, 64, 9, 5]
            DOConv2d(64, 64, kernel_size=3, stride=2, padding=1),  # x = [50670, 64, 9, 5]
            nn.ReLU(),
            # ASFF(level=0),
            nn.Dropout2d(0.3),

            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),  # x = [50670, 32, 9, 5]
            nn.AdaptiveAvgPool2d((2, 2))            # [50670, 32, 2, 2]
        )
        self.linear = nn.Linear(in_features=32*2*2, out_features=64)
        self.gru = nn.GRU(input_size=64, hidden_size=32, num_layers=2, batch_first=True) # [batch, input_size, -]
        self.linear1 = nn.Linear(32*6, 120)
        # self.linear1 = nn.Linear(64, 192)
        self.dropout = nn.Dropout(0.4)
        self.linear2 = nn.Linear(120, num_classes)

    # pytorch的网络层参数初始化
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # x分离连续16个三维图 [batch, 16, 5, 6, 9]

        x1 = torch.squeeze(x[:, 0, :, :, :], 1)  # [batch, 5, 6, 9]
        x2 = torch.squeeze(x[:, 1, :, :, :], 1)
        x3 = torch.squeeze(x[:, 2, :, :, :], 1)
        x4 = torch.squeeze(x[:, 3, :, :, :], 1)
        x5 = torch.squeeze(x[:, 4, :, :, :], 1)
        x6 = torch.squeeze(x[:, 5, :, :, :], 1)
        # x7 = torch.squeeze(x[:, 6, :, :], 1)
        # x8 = torch.squeeze(x[:, 7, :, :], 1)
        # x9 = torch.squeeze(x[:, 8, :, :], 1)
        # x10 = torch.squeeze(x[:, 9, :, :], 1)
        # x11 = torch.squeeze(x[:, 10, :, :], 1)
        # x12 = torch.squeeze(x[:, 11, :, :], 1)
        # x13 = torch.squeeze(x[:, 12, :, :], 1)
        # x14 = torch.squeeze(x[:, 13, :, :], 1)
        # x15 = torch.squeeze(x[:, 14, :, :], 1)
        # x16 = torch.squeeze(x[:, 15, :, :], 1)
        x1 = self.Atten(x1)  # [batch, 5, 6, 9]
        x2 = self.Atten(x2)
        x3 = self.Atten(x3)
        x4 = self.Atten(x4)
        x5 = self.Atten(x5)
        x6 = self.Atten(x6)

        # bneck模块
        x1 = self.bneck(x1)
        x2 = self.bneck(x2)
        x3 = self.bneck(x3)
        x4 = self.bneck(x4)
        x5 = self.bneck(x5)
        x6 = self.bneck(x6)
        # x7 = self.bneck(x7)
        # x8 = self.bneck(x8)
        # x9 = self.bneck(x9)
        # x10 = self.bneck(x10)
        # x11 = self.bneck(x11)
        # x12 = self.bneck(x12)
        # x13 = self.bneck(x13)
        # x14 = self.bneck(x14)
        # x15 = self.bneck(x15)
        # x16 = self.bneck(x16)

        # x1 = self.linear((50670, 1, 32*2*2))
        x1 = self.linear(x1.view(x1.shape[0], 1, -1))  # [50670, 1, 32*2*2] -> [50670, 1, 64]
        x2 = self.linear(x2.view(x2.shape[0], 1, -1))
        x3 = self.linear(x3.view(x3.shape[0], 1, -1))
        x4 = self.linear(x4.view(x4.shape[0], 1, -1))
        x5 = self.linear(x5.view(x5.shape[0], 1, -1))
        x6 = self.linear(x6.view(x6.shape[0], 1, -1))

        # 16个3d图分别卷积后连接 16个[batch, 1, 32] -> [batch, 16, 32]
        # out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16), dim=1)
        out = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)  # [50670, 6, 64]
        # LSTM后输出           [batch, 16, 120]
        out, (h, c) = self.gru(out)  # [50670, 6, 64] -> [50670, 6, 32]

        # 展开以放入线性层       [batch, 16*120]
        out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])  # [50670, 6, 32] -> [50670, 192]

        # 第一个线性层           [batch, 120]
        out = self.linear1(out)
        out = self.dropout(out)
        # 第二个线性层           [batch, ]
        out = self.linear2(out)
        # 结果输出              [batch, 1]
        return out

if __name__ == '__main__':
      a = torch.randn((5067, 6, 8, 9, 5))
      mynet = EARCNN(3)
      # net = depthwise_separable_conv(1, 32, 3)
      output = mynet(a)
      print(output)

