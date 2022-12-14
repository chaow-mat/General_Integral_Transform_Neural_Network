"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import os
import sys
sys.path.append('../nn')
from model_ol import *
from utility_ol import UnitGaussianNormalizer
from Adam import Adam
from tensorboardX import SummaryWriter
from scipy.io import savemat
import statistics
import numpy as np
import torch
from timeit import default_timer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--modes', type=int, default=12, help='')
parser.add_argument('--width', type=int, default=32)
parser.add_argument('--M',  type=int, default=2500, help="number of dataset")
parser.add_argument('--device', type=int, default=1, help="index of cuda device")
parser.add_argument('--state', type=str, default='train', help="evaluation ot training")
parser.add_argument('--noliz', type=bool, default=True)
parser.add_argument('--path_model', type=str, default='', help="path of model for testing")
cfg = parser.parse_args()
print(sys.argv)

device = torch.device('cuda:' + str(cfg.device))

batch_size = 64
M = cfg.M
width = cfg.width
modes = cfg.modes
N = 64
ntrain = M
ntest = M
s = N
learning_rate = 0.001
epochs = 5000
step_size = 500
gamma = 0.5

################################################################
# load data
################################################################
prefix = "/home/wangchao/dataset/FtF/"
# prefix = "dataset/"
data = np.load(prefix + "NavierStokes_40000_compressed.npz")
cs = data['inputs']
K = data['outputs']

# transpose
cs = cs.transpose(2, 0, 1)
K = K.transpose(2, 0, 1)

x_train = torch.from_numpy(np.reshape(cs[:ntrain, :, :], -1).astype(np.float32))
y_train = torch.from_numpy(np.reshape(K[:ntrain, :, :], -1).astype(np.float32))

x_test = torch.from_numpy(np.reshape(cs[ntrain:ntrain+ntest, :, :], -1).astype(np.float32))
y_test = torch.from_numpy(np.reshape(K[ntrain:ntrain+ntest, :, :], -1).astype(np.float32))

x_normalizer = UnitGaussianNormalizer(x_train, device, cfg.noliz)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train, device, cfg.noliz)
y_train = y_normalizer.encode(y_train)
y_test = y_normalizer.encode(y_test)

x_train = x_train.reshape(ntrain,s,s,1)
y_train = y_train.reshape(ntrain,s,s,1)
x_test = x_test.reshape(ntest,s,s,1)
y_test = y_test.reshape(ntest,s,s,1)

################################################################
# training and evaluation
################################################################

string =  str(ntrain) + "_cw" + str(width) + "_m" + str(modes) + "_lr" + str(learning_rate) + "-" + str(step_size) + "-" + str(gamma) +  '_noliz' + str(cfg.noliz)

if cfg.state=='train':
    path = "training/FNO/FNO_"+string
    if not os.path.exists(path):
        os.makedirs(path)
    writer = SummaryWriter(log_dir=path)
    model = FNO2d(modes, modes, width).to(device)
    path_model = "model/FNO/"
    if not os.path.exists(path_model):
        os.makedirs(path_model)

else: # eval
    if (cfg.path_model):
        model = torch.load(cfg.path_model, map_location=device)
    else:
        model = torch.load("model/FNO/FNO_" + string + ".model", map_location=device)
    epochs = 1
    batch_size = 1

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
x_normalizer.cuda()
t0 = default_timer()
for ep in range(epochs):
    if cfg.state=='train':
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x).reshape(batch_size_, -1)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            loss = myloss(out.view(batch_size_,-1), y)
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()

        train_l2/= ntrain

        writer.add_scalar("train/error", train_l2, ep)

    average_relative_error = 0
    error_list = []
    with torch.no_grad():
        for x, y, in test_loader:
            x, y = x.to(device), y.to(device)
            batch_size_ = x.shape[0]
            out = model(x).reshape(batch_size_, -1)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            if ep % 10 == 0:
                y = y.reshape(batch_size_, -1)
                norms = torch.norm(y, dim=1)
                error = y - out
                relative_error = torch.norm(error, dim=1) / norms
                error_list.append(relative_error)
                average_relative_error += torch.sum(relative_error)
    if ep % 10 == 0:
        average_relative_error = average_relative_error / (ntest)
        print(f"Average Relative Test Error : {ep }{average_relative_error: .6e} ")
        if cfg.state=='train':
            writer.add_scalar("test/error", average_relative_error, ep)

    if cfg.state=='eval':
        median = statistics.median(error_list)
        idx_median = min(range(len(error_list)), key=lambda i: abs(error_list[i] - median))
        maxx = max(error_list)
        idx_max = min(range(len(error_list)), key=lambda i: abs(error_list[i] - maxx))
        # median
        input = x_normalizer.decode(x_test[idx_median:idx_median+1, :, :, :].to(device)).reshape(1, -1).detach().cpu().numpy()
        output = y_normalizer.decode(model(x_test[idx_median:idx_median+1, :, :, :].to(device))).reshape(1, -1).detach().cpu().numpy()
        output_true = y_normalizer.decode(y_test[idx_median:idx_median+1, :, :, :].to(device)).reshape(1, -1).detach().cpu().numpy()
        savemat('predictions/FNO/FNO_ns_median_' + string + '_id' + str(idx_median) +'.mat', {'input':input, 'output': output, 'output_true':output_true})
        # max
        input = x_normalizer.decode(x_test[idx_max:idx_max + 1, :, :, :].to(device)).reshape(1, -1).detach().cpu().numpy()
        output = y_normalizer.decode(model(x_test[idx_max:idx_max + 1, :, :, :].to(device))).reshape(1, -1).detach().cpu().numpy()
        output_true = y_normalizer.decode(y_test[idx_max:idx_max+1, :, :, :].to(device)).reshape(1, -1).detach().cpu().numpy()
        savemat('predictions/FNO/FNO_ns_max_' + string + '_id' + str(idx_max) + '.mat',
                {'input': input, 'output': output, 'output_true': output_true})

if cfg.state=='train':
    torch.save(model.state_dict(), 'model/FNO/FNO_' + string + '.model')