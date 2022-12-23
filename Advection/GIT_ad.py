import os
import sys
import numpy as np
import pylab as plt
# import sklearn
import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
sys.path.append('../nn')
from model_ol import *
from utility_ol import MinMaxNormalizer
from Adam import Adam
import argparse
import sys
from scipy.io import savemat
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--c_width', type=int, default=16, help='')
parser.add_argument('--d_width', type=int, default=512)
parser.add_argument('--M',  type=int, default=2500, help="number of dataset")
parser.add_argument('--dim_PCA', type=int, default=200)
parser.add_argument('--eps', type=float, default=1e-6)
parser.add_argument('--noliz', type=bool, default=True)
parser.add_argument('--device', type=int, default=0, help="index of cuda device")
parser.add_argument('--state', type=str, default='train')
parser.add_argument('--path_model', type=str, default='', help="path of model for testing")
cfg = parser.parse_args()

print(sys.argv)
device = torch.device('cuda:' + str(cfg.device))

# parameters
ntrain = cfg.M
ntest = cfg.M
layer = 3
c_width = cfg.c_width
d_width = cfg.d_width
batch_size = 64
learning_rate = 0.0001
num_epoches = 3000
ep_predict = 10
step_size = 100
gamma = 0.5

# load data
prefix = "~/dataset/FtF/"
data = np.load(prefix + "Advection_40000_compressed.npz")
inputs = data['inputs']
outputs = data['outputs']

# PCA
train_inputs = np.reshape(inputs[:,  :ntrain], (-1, ntrain))
test_inputs = np.reshape(inputs[:, ntrain:ntrain + ntest], (-1, ntest))
Ui, Si, Vi = np.linalg.svd(train_inputs)
en_f = 1 - np.cumsum(Si) / np.sum(Si)
r_f = np.argwhere(en_f < cfg.eps)[0, 0]
if r_f>cfg.dim_PCA:
    r_f = cfg.dim_PCA

Uf = Ui[:, :r_f]
f_hat = np.matmul(Uf.T, train_inputs)
f_hat_test = np.matmul(Uf.T, test_inputs)
x_train = torch.from_numpy(f_hat.T.astype(np.float32))
x_test = torch.from_numpy(f_hat_test.T.astype(np.float32))

train_outputs = np.reshape(outputs[:,  :ntrain], (-1, ntrain))
test_outputs = np.reshape(outputs[:,  ntrain:ntrain+ntest], (-1, ntest))
Uo, So, Vo = np.linalg.svd(train_outputs)
en_g = 1 - np.cumsum(So) / np.sum(So)
r_g = np.argwhere(en_g < cfg.eps)[0, 0]
if r_g>cfg.dim_PCA:
    r_g = cfg.dim_PCA
Ug = Uo[:, :r_g]
g_hat = np.matmul(Ug.T, train_outputs)
g_hat_test = np.matmul(Ug.T, test_outputs)
y_train = torch.from_numpy(g_hat.T.astype(np.float32))
y_test = torch.from_numpy(g_hat_test.T.astype(np.float32))
test_outputs = torch.from_numpy(test_outputs).to(device)

# normalization
x_normalizer = MinMaxNormalizer(x_train, -1, 1, device, cfg.noliz)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = MinMaxNormalizer(y_train, -1, 1, device, cfg.noliz)
y_train = y_normalizer.encode(y_train)
y_test = y_normalizer.encode(y_test)
y_normalizer.cuda()

print("Input #bases : ", r_f, " output #bases : ", r_g)

model = GIT(r_f, d_width, c_width, r_g)
string = str(ntrain) + '_dpca_' + str(r_f) + '-' + str(r_g) + '_l' + str(layer) + '_act_gelu' + '_dw' + str(d_width) + '_cw' + str(c_width) + '_lr' + str(learning_rate) + '-' + str(step_size) + '-' + str(gamma)+ '_noliz' + str(cfg.noliz)
# path to save model
if cfg.state=='train':
    path = 'training/GIT/GIT_' + string
    if not os.path.exists(path):
        os.makedirs(path)
    writer = SummaryWriter(log_dir=path)

    path_model = "model/GIT/"
    if not os.path.exists(path_model):
        os.makedirs(path_model)
else:
    if (cfg.path_model):
        model_state_dict = torch.load(cfg.path_model, map_location=device)
        model.load_state_dict(model_state_dict)
    else:
        model_state_dict = torch.load('model/GIT/GIT_' + string +  '.model', map_location=device)
        model.load_state_dict(model_state_dict)
    num_epoches = 1
    batch_size = 1

# data loader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, test_outputs.T),batch_size=batch_size, shuffle=False)

model = model.float()

if torch.cuda.is_available():
    model = model.to(device)

# model loss
criterion = LpLoss(size_average=False)
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# TRAINING
for ep in range(num_epoches):
    if cfg.state=='train':
        model.train()
        train_l2_step = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            loss = criterion(out, y)
            train_l2_step += loss.item()

            #  backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        train_l2_step /= ntrain
        writer.add_scalar("train/error", train_l2_step, ep)

# Validation
    test_l2_step = 0
    average_relative_error = 0
    error_list = []
    with torch.no_grad():
        for x, _, y_test in test_loader:
            x= x.to(device)
            out = model(x)
            out = y_normalizer.decode(out).detach().cpu().numpy()
            if ep % ep_predict == 0:
                y_test = y_test.detach().cpu().numpy()
                y_test_pred = np.matmul(Ug, out.T)
                norms = np.linalg.norm(y_test, axis=1)
                error = y_test - y_test_pred.T
                relative_error = np.linalg.norm(error, axis=1) / norms
                error_list.append(relative_error)
                average_relative_error += np.sum(relative_error)

    if ep % ep_predict == 0:
        average_relative_error = average_relative_error / ntest
        print(f"Average Relative Error of original PCA: {ep } {average_relative_error: .6e}")
        if cfg.state=='train':
            writer.add_scalar("test/error", average_relative_error, ep)

    if cfg.state=='eval':
        median = statistics.median(error_list)
        idx_median = min(range(len(error_list)), key=lambda i: abs(error_list[i] - median))
        maxx = max(error_list)
        idx_max = min(range(len(error_list)), key=lambda i: abs(error_list[i] - maxx))
        # median
        input = test_inputs[:, idx_median:idx_median + 1].reshape(-1, 1)
        output = y_normalizer.decode(model(x_test[idx_median:idx_median + 1, :].to(device))).reshape(1, -1).detach().cpu().numpy()
        output = np.matmul(Ug, output.T).T.reshape(-1, 1)
        output_true = test_outputs[:, idx_median:idx_median + 1].reshape(-1, 1).detach().cpu().numpy()
        savemat('predictions/GIT/GIT_ad_median_' + string + '_id' + str(idx_median) + '.mat',
                {'input': input, 'output': output, 'output_true': output_true})
        # max
        input = test_inputs[:, idx_max:idx_max + 1].reshape(-1, 1)
        output = y_normalizer.decode(model(x_test[idx_max:idx_max + 1, :].to(device))).reshape(1, -1).detach().cpu().numpy()
        output = np.matmul(Ug, output.T).T.reshape(-1, 1)
        output_true = test_outputs[:, idx_max:idx_max + 1].reshape(-1, 1).detach().cpu().numpy()
        savemat('predictions/GIT/GIT_ad_max_' + string + '_id' + str(idx_max) + '.mat',
                {'input': input, 'output': output, 'output_true': output_true})

# save model
if cfg.state=='train':
    torch.save(model.state_dict(), 'model/GIT/GIT_' + string + '.model')




