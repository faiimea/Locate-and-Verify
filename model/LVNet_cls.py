import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("..")
# Here import the srm_conv!!
from components.attention import ChannelAttention, SpatialAttention
from components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
from components.linear_fusion import HdmProdBilinearFusion
from model.xception import TransferModel
from model.modules import *


class Two_Stream_Net_Cls(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = {
            'location': {
                'size': 19,
                'channels': [64, 128, 256, 728, 728, 728],
                'mid_channel': 512
            },
            'cls_size': 10,
            'HBFusion': {
                'hidden_dim': 2048,
                'output_dim': 4096,
            }
        }
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.xception_srm = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)

        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.relu = nn.ReLU(inplace=True)

        self.cmc0 = CMCE(in_channel=self.params['location']['channels'][0])
        self.cmc1 = CMCE(in_channel=self.params['location']['channels'][1])
        self.cmc2 = CMCE(in_channel=self.params['location']['channels'][2])

        self.lfe0 = LFGA(in_channel=self.params['location']['channels'][3])
        self.lfe1 = LFGA(in_channel=self.params['location']['channels'][4])
        self.lfe2 = LFGA(in_channel=self.params['location']['channels'][5])

        self.HBFusion = HdmProdBilinearFusion(dim1=(64+128+256+728+728), dim2=2048, 
                        hidden_dim=self.params['HBFusion']['hidden_dim'], output_dim=self.params['HBFusion']['output_dim'])

        self.cls_header = nn.Sequential(
            nn.BatchNorm2d(self.params['HBFusion']['output_dim']),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(self.params['HBFusion']['output_dim'], 2),
        )

    def pad_max_pool(self, x):
        b, c, h, w = x.size()
        padding = abs(h % self.params['cls_size'] - self.params['cls_size']) % self.params['cls_size'] 
        pad = nn.ReplicationPad2d(padding=(padding // 2, (padding + 1) // 2, padding // 2, (padding + 1) // 2)).to(x.device)
        x = pad(x)
        b, c, h, w = x.size()
        
        max_pool = nn.MaxPool2d(kernel_size=h // self.params['cls_size'], stride=h // self.params['cls_size'], padding=0)
        return max_pool(x)

    def features(self, x):
        srm = self.srm_conv0(x)

        # 64 * 150 * 150
        x0 = self.xception_rgb.model.fea_part1_0(x)
        y0 = self.xception_srm.model.fea_part1_0(srm)
        x0, y0 = self.cmc0(x0, y0)

        # 128 * 75 * 75
        x1 = self.xception_rgb.model.fea_part1_1(x0)
        y1 = self.xception_srm.model.fea_part1_1(y0)
        x1, y1 = self.cmc1(x1, y1)

        # 256 * 38 * 38
        x2 = self.xception_rgb.model.fea_part1_2(x1)
        y2 = self.xception_srm.model.fea_part1_2(y1)
        x2, y2 = self.cmc2(x2, y2)

        # 728 * 19 * 19
        x3 = self.xception_rgb.model.fea_part1_3(x2+y2)
        y3 = self.xception_srm.model.fea_part1_3(x2+y2)
        y3 = self.lfe0(y3, x3)

        # 728 * 19 * 19
        x4 = self.xception_rgb.model.fea_part2_0(x3)
        y4 = self.xception_srm.model.fea_part2_0(y3)
        y4 = self.lfe1(y4, x4)

        # 728 * 19 * 19
        x5 = self.xception_rgb.model.fea_part2_1(x4)
        y5 = self.xception_srm.model.fea_part2_1(y4)
        y5 = self.lfe2(y5, x5)
        
        # 2048 * 10 * 10
        x6 = self.xception_rgb.model.fea_part3(x5)
        y6 = self.xception_srm.model.fea_part3(y5)

        # Multi-stream fusion
        y0m = self.pad_max_pool(y0)
        y1m = self.pad_max_pool(y1)
        y2m = self.pad_max_pool(y2)
        y3m = self.pad_max_pool(y3)
        y5m = self.pad_max_pool(y5)
        mul_feas = torch.cat((y0m, y1m, y2m, y3m, y5m), dim=1)
        # print(f"mul_feas size: {mul_feas.size()}")
        # print(f"y6 size: {y6.size()}")
        y6_resized = F.interpolate(y6, size=(mul_feas.size(2), mul_feas.size(3)), mode='bilinear', align_corners=False)
        cls_feas = self.HBFusion(mul_feas, y6_resized)

        return cls_feas

    def forward(self, x):
        # Extract features and classify
        cls_feas = self.features(x)
        cls_preds = self.cls_header(cls_feas)
        return cls_preds


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
