import torch
import sys
import re
sys.path.append('../nn')
import torch.nn as nn
import torch.nn.functional as F

class fcn(nn.Module):
    def __init__(self, in_dim, d_width, c_width, out_dim):
        super(fcn, self).__init__()
        self.in_dim = in_dim
        self.d_width = d_width
        self.c_width = c_width
        self.lift_c = nn.Linear(1, c_width)
        self.lift_d = nn.Linear(in_dim, d_width)

        self.layer1_c = nn.Linear(c_width, c_width)
        self.layer2_c = nn.Linear(c_width, c_width)
        self.layer3_c = nn.Linear(c_width, c_width)

        self.layer1_d1 = nn.Linear(d_width, d_width)
        self.layer2_d1 = nn.Linear(d_width, d_width)
        self.layer3_d1 = nn.Linear(d_width, d_width)

        self.layer1_d2 = nn.Linear(d_width, d_width)
        self.layer2_d2 = nn.Linear(d_width, d_width)
        self.layer3_d2 = nn.Linear(d_width, d_width)

        self.layer4_c = nn.Linear(c_width, 1)
        self.layer4_d = nn.Linear(d_width, out_dim)
        # self.act = nn.gelu()

        self.scale = (1 / (c_width * c_width))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(c_width, c_width, d_width, dtype=torch.float))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(c_width, c_width, d_width, dtype=torch.float))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(c_width, c_width, d_width, dtype=torch.float))



    def forward(self, x):
        b = x.size(0)
        x = x.unsqueeze(2) # (b, nx, c=1)
        x = self.lift_c(x) # (b, nx, c=width)
        x = x.permute(0, 2, 1)  # (b, c, nx)
        x = self.lift_d(x)
        x = x.permute(0, 2, 1)  # (b, nx. c)


        x1 = self.layer1_c(x) # (b, nx, c)
        x2 = x.permute(0, 2, 1) # (b, c, nx)
        x2 = self.layer1_d1(x2)
        x2 = torch.einsum("bix,iox->box", x2, self.weights1)
        x2 = self.layer1_d2(x2)
        x2 = x2.permute(0, 2, 1)  # (b, nx, c)
        x = x1 + x2
        # x = self.act(x)
        x = F.gelu(x)

        x1 = self.layer2_c(x) # (b, nx, c)
        x2 = x.permute(0, 2, 1) # (b, c, nx)
        x2 = self.layer2_d1(x2)
        x2 = torch.einsum("bix,iox->box", x2, self.weights2)
        x2 = self.layer2_d2(x2)
        x2 = x2.permute(0, 2, 1)  # (b, nx, c)
        x = x1 + x2
        # x = self.act(x)
        x = F.gelu(x)

        x1 = self.layer3_c(x) # (b, nx, c)
        x2 = x.permute(0, 2, 1) # (b, c, nx)
        x2 = self.layer3_d1(x2)
        x2 = torch.einsum("bix,iox->box", x2, self.weights3)
        x2 = self.layer3_d2(x2)
        x2 = x2.permute(0, 2, 1)  # (b, nx, c)
        x = x1 + x2

        x = self.layer4_c(x)
        x = x.permute(0, 2, 1)  # (b, c, nx)
        x = self.layer4_d(x)

        x = x.squeeze(1)

        return x


# string = 'FNO_10000_cw4_m12_lr0.001-100-0.5_nolizTrue-lploss-orinoliz',
# list = os.listdir('model/PCA') # to get the list of all files
string = ['PodDeepOnet_10000_dpca_115_l3_dw16_cw2_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_5000_dpca_114_l3_dw64_cw4_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_20000_dpca_115_l3_dw128_cw8_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_2500_dpca_279_l3_dw256_cw16_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_5000_dpca_114_l3_dw256_cw16_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_2500_dpca_279_l3_dw64_cw4_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_20000_dpca_115_l3_dw16_cw2_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_10000_dpca_115_l3_dw512_cw32_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_10000_dpca_115_l3_dw256_cw16_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_2500_dpca_279_l3_dw512_cw32_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_20000_dpca_115_l3_dw256_cw16_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_10000_dpca_115_l3_dw64_cw4_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_20000_dpca_115_l3_dw64_cw4_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_5000_dpca_114_l3_dw16_cw2_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_5000_dpca_114_l3_dw128_cw8_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_10000_dpca_115_l3_dw128_cw8_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_2500_dpca_279_l3_dw128_cw8_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_5000_dpca_114_l3_dw512_cw32_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_20000_dpca_115_l3_dw512_cw32_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model', 'PodDeepOnet_2500_dpca_279_l3_dw16_cw2_lw4096_lr_0.001-500-0.5_nolizTrue-kernel5-lploss-gaunoliz.model']


# FNO
# for i in range(len(string)):
#     num = re.findall(r'\d+', string[i])
#     state = torch.load('model/FNO/'+string[i],map_location='cuda:0')
#     torch.save(state, 'model/FNO/FNO_' + num[0] + '_cw' + num[1] + '.model')


# PCA
# for i in range(len(string)):
#     num = re.findall(r'\d+', string[i])
#     state = torch.load('model/PCA/'+string[i],map_location='cuda:0')
#     torch.save(state, 'model/PCA/PCA_' + num[0] + '_cw' + num[3] + '.model')


# GIT
# for i in range(len(string)):
#     num = re.findall(r'\d+', string[i])
#     state = torch.load('model/GIT/'+string[i],map_location='cuda:0')
#     torch.save(state, 'model/GIT/GIT_' + num[0] + '_dw' + num[4] + '_cw' + num[5] + '.model')

# POD
for i in range(len(string)):
    num = re.findall(r'\d+', string[i])
    state = torch.load('model/PodDeepOnet/'+string[i],map_location='cuda:0')
    torch.save(state, 'model/PodDeepOnet/PodDeepOnet_' + num[0] + '_dw' + num[3] + '_cw' + num[4] + '.model')