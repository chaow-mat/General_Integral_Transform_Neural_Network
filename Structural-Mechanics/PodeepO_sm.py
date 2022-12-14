import os
import sys
import numpy as np
import scipy.io as io
import torch

sys.path.append('../nn')
from model_ol import *
from utility_ol import UnitGaussianNormalizer
from Adam import Adam
from timeit import default_timer
import argparse
from tensorboardX import SummaryWriter
from scipy.io import savemat
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--M', type=int, default=2500, help="number of dataset")
parser.add_argument('--lift_width', type=int, default=3136)
parser.add_argument('--c_width', type=int, default=32)
parser.add_argument('--d_width', type=int, default=512)
parser.add_argument('--layers', type=int, default=3, help='layers of CNN')
parser.add_argument('--device', type=int, default=2)
parser.add_argument('--dim_PCA', type=int, default=200)
parser.add_argument('--eps', type=float, default=1e-3)
parser.add_argument('--noliz', type=bool, default=True)
parser.add_argument('--state', type=str, default='train')
parser.add_argument('--path_model', type=str, default='', help="path of model for testing")
cfg = parser.parse_args()
print(sys.argv)

device = torch.device('cuda:' + str(cfg.device))

M = cfg.M
batch_size = int(M/2500 * 128)
N = 101
ntrain = cfg.M
ntest = cfg.M
N_theta = 100
learning_rate = 0.001
epochs = 2000
step_size = 500
gamma = 0.5

# load data
prefix = "/home/wangchao/dataset/FtF/"
# prefix = "dataset/"
data = io.loadmat(prefix + "/Static-planestress_40000")

inputs = data['f_bc'] # [40000, 101]
outputs = data['stress']  # [40000, 4072]

x_train = torch.from_numpy(inputs[:ntrain, :].astype(np.float32).reshape(-1))
x_test = torch.from_numpy(inputs[ntrain:ntrain+ntest, :].astype(np.float32).reshape(-1))
y_train = torch.from_numpy(outputs[:ntrain, :].astype(np.float32).reshape(-1))
y_test = torch.from_numpy(outputs[ntrain:ntrain+ntest, :].astype(np.float32).reshape(-1))

# normalization
x_normalizer = UnitGaussianNormalizer(x_train, device, cfg.noliz)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = UnitGaussianNormalizer(y_train, device, cfg.noliz)
y_train = y_normalizer.encode(y_train)
y_test = y_normalizer.encode(y_test)
y_normalizer.cuda()

x_train = x_train.reshape(ntrain, 101)
y_train = y_train.reshape(ntrain, 4072)
x_test = x_test.reshape(ntest, 101)
y_test = y_test.reshape(ntest, 4072)

# compute trunk basis
Uo, So, Vo = np.linalg.svd(y_train.T)
en_g = 1 - np.cumsum(So) / np.sum(So)
r_g = np.argwhere(en_g < cfg.eps)[0, 0]
# r_g = dim_PCA
if r_g > cfg.dim_PCA:
    r_g = cfg.dim_PCA
trunk_basis = torch.from_numpy(np.sqrt(Uo.shape[0]) * Uo[:, :r_g].astype(np.float32)).to(device)


print(" output #bases : ", r_g)

################################################################
# training and evaluation
################################################################
string =  str(ntrain) + '_dpca_' + str(r_g) + '_l' + str(cfg.layers) + '_dw' + str(cfg.d_width) + '_cw' + str(cfg.c_width) + '_lw' + str(cfg.lift_width) + '_lr_' + str(learning_rate) + '-' + str(step_size) + '-' + str(gamma) + '_noliz' + str(cfg.noliz)

if cfg.state=='train':
    path = 'training/PodDeepOnet/PodDeepOnet_' + string
    if not os.path.exists(path):
        os.makedirs(path)
    writer = SummaryWriter(log_dir=path)
    c_width = [cfg.c_width] * cfg.layers
    kernels = [3] * cfg.layers
    h = int(np.sqrt(cfg.lift_width))
    w = h
    model = PodDeepOnet_2d_unstd(trunk_basis, 101, cfg.lift_width, h, w, cfg.d_width, cfg.layers, c_width, kernels, r_g, 'relu')
    path_model = "model/PodDeepOnet/"
    if not os.path.exists(path_model):
        os.makedirs(path_model)
else:
    if (cfg.path_model):
        model = torch.load(cfg.path_model, map_location=device)
    else:
        model = torch.load('model/PodDeepOnet/PodDeepOnet_' + string +  '.model', map_location=device)
    epochs = 1
    batch_size = 1

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=batch_size, shuffle=False)

model.to(device)

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)

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
            out = model(x)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            loss = myloss(out, y)
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()

        train_l2 /= ntrain

        t2 = default_timer()
        writer.add_scalar("train/loss", train_l2, ep)
        # print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)

    average_relative_error = 0
    error_list = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            out = y_normalizer.decode(out).detach().cpu().numpy()
            y = y_normalizer.decode(y).cpu().numpy()
            if ep % 10 == 0:
                norms = np.linalg.norm(y, axis=1)
                error = y - out
                relative_error = np.linalg.norm(error, axis=1) / norms
                error_list.append(relative_error)
                average_relative_error += np.sum(relative_error)

    if ep % 10 == 0:
        average_relative_error = average_relative_error / (ntest)
        print(f"Average Relative Test Error of PCA: {ep} {average_relative_error: .6e}")
        if cfg.state=='train':
            writer.add_scalar("test/error", average_relative_error, ep)

    if cfg.state == 'eval':
        median = statistics.median(error_list)
        idx_median = min(range(len(error_list)), key=lambda i: abs(error_list[i] - median))
        maxx = max(error_list)
        idx_max = min(range(len(error_list)), key=lambda i: abs(error_list[i] - maxx))
        # median
        input = inputs[ntrain+idx_median:ntrain+idx_median + 1, :].reshape(1, -1)
        output = y_normalizer.decode(model(x_test[idx_median:idx_median + 1, :].to(device))).reshape(1, -1).detach().cpu().numpy()
        output_true = outputs[ntrain+idx_median:ntrain+idx_median + 1, :].reshape(1, -1)
        savemat('predictions/PodDeepOnet/PodDeepOnet_sm_median_' + string + '_id' + str(idx_median) + '.mat',
                {'input': input, 'output': output, 'output_true': output_true})
        # max
        input = inputs[ntrain + idx_max:ntrain + idx_max + 1, :].reshape(1, -1)
        output = y_normalizer.decode(model(x_test[idx_max:idx_max + 1, :].to(device))).reshape(1, -1).detach().cpu().numpy()
        output_true = outputs[ntrain + idx_max:ntrain + idx_max + 1, :].reshape(1, -1)
        savemat('predictions/PodDeepOnet/PodDeepOnet_sm_max_' + string + '_id' + str(idx_max) + '.mat',
                {'input': input, 'output': output, 'output_true': output_true})

# save model
if cfg.state == 'train':
    torch.save(model.state_dict(), 'model/PodDeepOnet/PodDeepOnet_' + string + '.model')



