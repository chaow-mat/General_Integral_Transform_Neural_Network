import os
import sys
import numpy as np
import scipy.io as io

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
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--M', type=int, default=2500, help="number of dataset")
parser.add_argument('--device', type=int, default=2)
parser.add_argument('--dim_PCA', type=int, default=200)
parser.add_argument('--eps', type=float, default=1e-6)
parser.add_argument('--noliz', type=bool, default=True)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--state', type=str, default='train')
parser.add_argument('--path_model', type=str, default='', help="path of model for testing")
cfg = parser.parse_args()

print(sys.argv)

device = torch.device('cuda:' + str(cfg.device))

M = cfg.M
N_neurons = cfg.width
layers = cfg.layers
batch_size = 256
N = 200
ntrain = M
ntest = M
N_theta = 100
learning_rate = 0.001
epochs = 5000
step_size = 500
gamma = 0.5

# load data
prefix = "~/dataset/FtF/"
data = io.loadmat(prefix + "Darcy_Triangular_40000.mat")

inputs = data['f_bc'].T
outputs = data['u_field'].T



train_inputs = inputs[:, :ntrain]
test_inputs = inputs[:, ntrain:ntrain + ntest]
Ui, Si, Vi = np.linalg.svd(train_inputs)
en_f = 1 - np.cumsum(Si) / np.sum(Si)
r_f = np.argwhere(en_f < cfg.eps)[0, 0]


Uf = Ui[:, :r_f]
f_hat = np.matmul(Uf.T, train_inputs)
f_hat_test = np.matmul(Uf.T, test_inputs)
x_train = torch.from_numpy(f_hat.T.astype(np.float32).reshape(-1))
x_test = torch.from_numpy(f_hat_test.T.astype(np.float32).reshape(-1))

train_outputs = outputs[:, :ntrain]
test_outputs = outputs[:, ntrain:ntrain + ntest]
Uo, So, Vo = np.linalg.svd(train_outputs)
en_g = 1 - np.cumsum(So) / np.sum(So)
r_g = np.argwhere(en_g < cfg.eps)[0, 0]
# r_g = dim_PCA
Ug = Uo[:, :r_g]
g_hat = np.matmul(Ug.T, train_outputs)
g_hat_test = np.matmul(Ug.T, test_outputs)
y_train = torch.from_numpy(g_hat.T.astype(np.float32).reshape(-1))
y_test = torch.from_numpy(g_hat_test.T.astype(np.float32).reshape(-1))
test_outputs = torch.from_numpy(test_outputs).to(device)

x_normalizer = UnitGaussianNormalizer(x_train, device, cfg.noliz)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = UnitGaussianNormalizer(y_train, device, cfg.noliz)
y_train = y_normalizer.encode(y_train)
y_test = y_normalizer.encode(y_test)
y_normalizer.cuda()

x_train = x_train.reshape(ntrain, r_f)
y_train = y_train.reshape(ntrain, r_g)
x_test = x_test.reshape(ntest, r_f)
y_test = y_test.reshape(ntest, r_g)

print("Input #bases : ", r_f, " output #bases : ", r_g)

################################################################
# training and evaluation
################################################################
model = FNN(r_f, r_g, layers, N_neurons)
string = str(ntrain) + '_dpca_' + str(r_f) + '-' + str(r_g) + '_cw'+ str(cfg.width)

if cfg.state=='train':
    path = 'training/PCA/PCA_'+ string
    if not os.path.exists(path):
        os.makedirs(path)
    writer = SummaryWriter(log_dir=path)

    path_model = "model/PCA/"
    if not os.path.exists(path_model):
        os.makedirs(path_model)
else:
    if (cfg.path_model):
        model_state_dict = torch.load(cfg.path_model, map_location=device)
        model.load_state_dict(model_state_dict)
    else:
        model_state_dict = torch.load('model/PCA/PCA_'+ string + '.model', map_location=device)
        model.load_state_dict(model_state_dict)
    epochs = 1
    batch_size = 1

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, test_outputs.T),
                                          batch_size=batch_size, shuffle=False)

model.to(device)

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = criterion = LpLoss(size_average=False)

t0 = default_timer()
for ep in range(epochs):
    if cfg.state == 'train':
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
        for x, y, y_test in test_loader:
            x, y = x.to(device), y.to(device)
            batch_size_ = x.shape[0]
            out = model(x)
            out = y_normalizer.decode(out).detach().cpu().numpy()
            y = y_normalizer.decode(y)
            if ep % 10 == 0:
                y_test = y_test.detach().cpu().numpy()
                y_test_pred = np.matmul(Ug, out.T)
                norms = np.linalg.norm(y_test, axis=1)
                error = y_test - y_test_pred.T
                relative_error = np.linalg.norm(error, axis=1) / norms
                if cfg.state == 'eval':
                    error_list.append(relative_error.item())
                average_relative_error += np.sum(relative_error)
    if ep % 10 == 0:
        average_relative_error = average_relative_error / (ntest)
        print(f"Average Relative Test Error of PCA: {ep} {average_relative_error: .6e}")
        if cfg.state == 'train':
            writer.add_scalar("test/error", average_relative_error, ep)


    if cfg.state == 'eval':
        median = statistics.median(error_list)
        idx_median = min(range(len(error_list)), key=lambda i: abs(error_list[i] - median))
        maxx = max(error_list)
        idx_max = min(range(len(error_list)), key=lambda i: abs(error_list[i] - maxx))
        print('median ', idx_median, ':', median, ' max ', idx_max, ':', maxx)
        # median
        input = test_inputs[:, idx_median:idx_median + 1].reshape(-1, 1)
        output = y_normalizer.decode(model(x_test[idx_median:idx_median + 1, :].to(device))).reshape(1, -1).detach().cpu().numpy()
        output = np.matmul(Ug, output.T).T.reshape(-1, 1)
        output_true = test_outputs[:, idx_median:idx_median + 1].reshape(-1, 1).detach().cpu().numpy()
        savemat('predictions/PCA/PCA_poi_median_' + string + '_id' + str(idx_median) + '.mat',
                {'input': input, 'output': output, 'output_true': output_true})
        # max
        input = test_inputs[:, idx_max:idx_max + 1].reshape(-1, 1)
        output = y_normalizer.decode(model(x_test[idx_max:idx_max + 1, :].to(device))).reshape(1, -1).detach().cpu().numpy()
        output = np.matmul(Ug, output.T).T.reshape(-1, 1)
        output_true = test_outputs[:, idx_max:idx_max + 1].reshape(-1, 1).detach().cpu().numpy()
        savemat('predictions/PCA/PCA_poi_max_' + string + '_id' + str(idx_max) + '.mat',
                {'input': input, 'output': output, 'output_true': output_true})

if cfg.state == 'train':
        torch.save(model.state_dict(), 'model/PCA/PCA_' + string + '.model')