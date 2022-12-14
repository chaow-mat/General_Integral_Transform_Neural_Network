import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class UnitGaussianNormalizer(object):
    def __init__(self, x, device, flag, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        copy_x = x
        self.mean = torch.mean(copy_x, 0)
        self.std = torch.std(copy_x, 0)
        self.eps = eps
        self.device = device
        self.flag = flag

    def encode(self, x):
        if self.flag:
            return (x - self.mean) / (self.std + self.eps)
        else:
            return x
    
    def encode_(self, x):

        if self.flag:
            x -= self.mean
            x /= (self.std + self.eps)
        else:
            x = x
        

    def decode(self, x, sample_idx=None):
        if self.flag:
            if sample_idx is None:
                std = self.std + self.eps # n
                mean = self.mean
            else:
                if len(self.mean.shape) == len(sample_idx[0].shape):
                    std = self.std[sample_idx] + self.eps  # batch*n
                    mean = self.mean[sample_idx]
                if len(self.mean.shape) > len(sample_idx[0].shape):
                    std = self.std[:,sample_idx]+ self.eps # T*batch*n
                    mean = self.mean[:,sample_idx]
            return (x * std) + mean
        else:
            return x

    def decode_(self, x, sample_idx=None):
        if self.flag:
            if sample_idx is None:
                std = self.std + self.eps # n
                mean = self.mean
            else:
                if len(self.mean.shape) == len(sample_idx[0].shape):
                    std = self.std[sample_idx] + self.eps  # batch*n
                    mean = self.mean[sample_idx]
                if len(self.mean.shape) > len(sample_idx[0].shape):
                    std = self.std[:,sample_idx]+ self.eps # T*batch*n
                    mean = self.mean[:,sample_idx]

            x *= std
            x += mean
        else:
            x = x

    def cuda(self):
        # self.mean = self.mean.cuda()
        # self.std = self.std.cuda()
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class MinMaxNormalizer(object):
    def __init__(self, x, l, u, device, flag, eps=0.0):
        super(MinMaxNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        copy_x = x.reshape(-1)
        self.min = torch.min(copy_x)
        self.max = torch.max(copy_x)
        self.l = l
        self.scale = (u - l) / (self.max - self.min)
        self.eps = eps
        self.device = device
        self.flag = flag

    def encode(self, x):
        if self.flag:
            return (x - self.min) * self.scale + self.l
        else:
            return x

    def encode_(self, x):

        if self.flag:
            x = (x - self.min) * self.scale + self.l
        else:
            x = x

    def decode(self, x, sample_idx=None):
        if self.flag:
            if sample_idx is None:
                min = self.min
                scale = self.scale
                l = self.l
            return ((x - l) / scale) + min
        else:
            return x

    def decode_(self, x, sample_idx=None):
        if self.flag:
            if sample_idx is None:
                min = self.min
                scale = self.scale
                l = self.l
            x -= self.l
            x /= scale
            x += min
        else:
            x = x

    def cuda(self):
        self.min = self.min.to(self.device)
        self.max = self.max.to(self.device)
        self.scale = self.scale.to(self.device)

    def cpu(self):
        self.min = self.min.cpu()
        self.max = self.max.cpu()
        self.scale = self.scale.cpu()