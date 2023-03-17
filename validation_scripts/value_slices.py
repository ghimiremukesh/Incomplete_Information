import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

import dataio, utils, training, loss_functions, modules_picnn, diff_operators, modules

import torch
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



# ckpt_path = '../experiment_scripts/logs/test_for_iros_1/checkpoints/model_final.pth'
ckpt_path = '../experiment_scripts/logs/training_w_noncvx/checkpoints/model_final.pth'
activation = 'relu'

# initialize the model
# model = modules_picnn.SingleBVPNet(in_features=4, out_features=1, type=activation, mode='mlp', hidden_features=32,
#                                    num_hidden_layers=3)
model = modules.SingleBVPNet(in_features=4, out_features=1, type=activation, mode='mlp', hidden_features=32,
                                   num_hidden_layers=3)
model.to(device)
checkpoint = torch.load(ckpt_path)
try:
    model_weights = checkpoint['model']
except:
    model_weights = checkpoint

model.load_state_dict(model_weights)
model.eval()

fig2 = plt.figure(figsize=(6, 6))

ax2 = fig2.gca(projection='3d')

fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))


# get the data
num_points = 10
x = torch.zeros((num_points, 3))
x[:, 0] = 1
x[:, 1] = torch.linspace(-1, 1, num_points)
x[:, 2] = 0
p_t = torch.linspace(0, 1, num_points).reshape(-1, 1)

t = 1*torch.ones((num_points*num_points, 1))
v = 0*torch.ones((num_points*num_points, 1))

coords = torch.cat((x, p_t), dim=1)
P = coords[:, -1].detach().cpu().numpy().squeeze()
X = coords[:, 1].detach().cpu().numpy().squeeze()
P, X = np.meshgrid(P, X)
coords_in = np.hstack((t, X.reshape(-1, 1), v, P.reshape(-1, 1)))
coords_in = torch.from_numpy(coords_in).reshape(1, 100, 4)
coords_in = {'coords': coords_in}

V_true = 1*((P * X) - ((1 - P) * X))

V_model = model(coords_in)['model_out'].detach().cpu().numpy().squeeze()

V_model = V_model.reshape(P.shape)

# plot the data

ax2.plot_surface(P, X, V_model, color='red') # model value at some t=0
ax2.plot_surface(P, X, V_true, color='green')

tT = ax3.contour(P, X, V_model, colors='red')
vT = ax3.contour(P, X, V_true, colors='green')
ax3.set_xlabel('$p$')
ax3.set_ylabel('$V$')

plt.clabel(tT, inline=1, fontsize=10)

plt.legend()





plt.show()






