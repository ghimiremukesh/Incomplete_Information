import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

import dataio, utils, training, loss_functions, modules_picnn, diff_operators

import torch
import numpy as np
import math
import random
from torch.utils.data import DataLoader
import configargparse
import scipy.io as spio

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ckpt_path = '../experiment_scripts/logs/4d_tests/checkpoints/model_final.pth'
# ckpt_path = '../logs/random_p_test_revisit/checkpoints/model_final.pth'
ckpt_path = '../logs/remove_curvature/checkpoints/model_final.pth'
activation = 'tanh'

# initialize the model
model = modules_picnn.SingleBVPNet(in_features=4, out_features=1, type=activation, mode='mlp', hidden_features=32,
                                   num_hidden_layers=3)
model.to(device)
checkpoint = torch.load(ckpt_path)
try:
    model_weights = checkpoint['model']
except:
    model_weights = checkpoint

model.load_state_dict(model_weights)
model.eval()

# generate data for plotting value landscape
num_points = 10
x = torch.zeros((num_points, 3))
x[:, 0] = 0.5
x[:, 1] = torch.linspace(-1, 1, num_points)
x[:, 2] = 0.5
p_t = torch.linspace(0, 1, num_points).reshape(-1, 1)

t = 0.5*torch.ones((num_points*num_points, 1))
v = 0.5*torch.ones((num_points*num_points, 1))

coords = torch.cat((x, p_t), dim=1)
coords_in = {'coords': coords}
value_approx = model(coords_in)['model_out'].detach().cpu().numpy()

value_true = -1*((coords[:, -1] * coords[:, 1]) - ((1 - coords[:, -1]) * coords[:, 1])).reshape(-1, 1).detach().cpu().numpy()

print()

# create figures
# fig_value = plt.figure(figsize=(6, 6))
# ax1 = fig_value.add_subplot(2, 1, 1)
# ax2 = fig_value.add_subplot(2, 1, 2)
# s = ax1.imshow(value_approx.T, cmap='bwr', origin='lower', extent=(0, 1., 0, 1.), vmin=0.25, vmax=0.2)
# fig_value.colorbar(s, ax=ax1)
# s2 = ax2.imshow(value_true.T, cmap='bwr', origin='lower', extent=(0, 1., 0, 1.), vmin=0.25, vmax=0.2)
# fig_value.colorbar(s2, ax=ax2)
# plt.plot(p_t.detach().cpu().numpy().squeeze(), value_true.squeeze())
# plt.show()

# draw value slices
# fig = plt.figure(figsize=(6, 6))
fig2 = plt.figure(figsize=(6, 6))
# ax = fig.gca(projection='3d')
ax2 = fig2.gca(projection='3d')

ax2.set_xlabel('$p$')
ax2.set_ylabel('$x$')
ax2.set_zlabel('$v$')

# get the data
P = coords[:, -1].detach().cpu().numpy().squeeze()
X = coords[:, 1].detach().cpu().numpy().squeeze()
P, X = np.meshgrid(P, X)
coords_in = np.hstack((t, X.reshape(-1, 1), v, P.reshape(-1, 1)))
coords_in = torch.from_numpy(coords_in).reshape(1, 100, 4)
coords_in = {'coords': coords_in}

V_true = -1*((P * X) - ((1 - P) * X))

model_output = model(coords_in)
y = model_output['model_out']
model_in = model_output['model_in']
grad, _ = diff_operators.jacobian(y, model_in)
dvdx = grad[..., 0, 1:-1].squeeze()  # exclude the last one
dvdt = grad[..., 0, 0].squeeze()
V_model = model(coords_in)['model_out'].detach().cpu().numpy().squeeze()

lam_d = dvdx[:, :1].squeeze()
lam_v = dvdx[:, -1:].squeeze()
# dvdt = grad[:, :, 0]

del_v = model_output['model_in'][:, :, 2].squeeze()

# action candidates
u_c = torch.tensor([-0.3, 0.3])
d_c = torch.tensor([-0.1, 0.1])
H = torch.zeros(100, 2, 2)

# for i in range(len(u_c)):
#     for j in range(len(d_c)):
#         H[:, i, j] = lam_da * v1 + lam_va * u_c[i] + lam_dd * v2 + lam_dv * d_c[j]
for i in range(len(u_c)):
    for j in range(len(d_c)):
        H[:, i, j] = lam_d * del_v + lam_v * (u_c[i] - d_c[j])

u = torch.zeros(100)
d = torch.zeros(100)
# pick action based on min_u max_d H
for i in range(100):
    d_index = torch.argmax(H[i, :, :], dim=1)[1]
    u_index = torch.argmin(H[i, :, d_index])
    u[i] = u_c[u_index]
    d[i] = d_c[d_index]

u = u.squeeze().to(device)
d = d.squeeze().to(device)

# ham = lam_da * v1 + lam_va * u + lam_dd * v2 + lam_dv * d
ham = lam_d * del_v + lam_v * (u - d)

V_model = V_model.reshape(P.shape)

# plot the data
# ax.plot_surface(P, X, V_true) # true value
# ax.set_title('True Value')
ax2.plot_surface(P, X, V_model, color='blue')
# surft._edgecolors2d = surft._edgecolor3d
# surft._facecolors2d = surft._facecolor3d

# model value
# dummyline1 = matplotlib.lines.Line2D([0],[0], linestyle="none", c='gray', marker = 'o')
# ax2.lenged([dummyline1], ['Value at t=0.8'])


fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
t0 = ax3.contour(P, X, V_model, colors='blue') # model value
plt.clabel(t0, inline=1, fontsize=10)
# t0.collections[0].set_label('Value at t=0.8')
# ax3.set_title('Model Value at t=0.8')

# fig4, ax4 = plt.subplots(1, 1, figsize=(6, 6))
# ax4.contour(P, X, V_true)
# ax4.set_title('True value')



### repeat for final time
# generate data for plotting value landscape
num_points = 10
x = torch.zeros((num_points, 3))
x[:, 0] = 0
x[:, 1] = torch.linspace(-1, 1, num_points)
x[:, 2] = 0
p_t = torch.linspace(0, 1, num_points).reshape(-1, 1)

t = 0*torch.ones((num_points*num_points, 1))
v = torch.zeros((num_points*num_points, 1))

coords = torch.cat((x, p_t), dim=1)
coords_in = {'coords': coords}
value_approx = model(coords_in)['model_out'].detach().cpu().numpy()

value_true = -1*((coords[:, -1] * coords[:, 1]) - ((1 - coords[:, -1]) * coords[:, 1])).reshape(-1, 1).detach().cpu().numpy()


# get the data
P = coords[:, -1].detach().cpu().numpy().squeeze()
X = coords[:, 1].detach().cpu().numpy().squeeze()
P, X = np.meshgrid(P, X)
coords_in = np.hstack((t, X.reshape(-1, 1), v, P.reshape(-1, 1)))
coords_in = torch.from_numpy(coords_in).reshape(1, 100, 4)
coords_in = {'coords': coords_in}

V_true = -1*((P * X) - ((1 - P) * X))

V_model = model(coords_in)['model_out'].detach().cpu().numpy().squeeze()

V_model = V_model.reshape(P.shape)

# plot the data
ax2.plot_surface(P, X, V_model, color='red') # model value
# surfT._edgecolors2d = surfT._edgecolor3d
# surfT._facecolors2d = surfT._facecolor3d

# ax2.lenged()
# fig6, ax6 = plt.subplots(1, 1, figsize=(6, 6))
tT = ax3.contour(P, X, V_model, colors='red')
ax3.set_xlabel('$p$')
ax3.set_ylabel('$V$')
# tT.set_label('Value at final time (T)')# model value
plt.clabel(tT, inline=1, fontsize=10)
# tT.collections[0].set_label('Model Value at final time (T)')
plt.legend()
# ax3.set_title(f'Model Value at T')




plt.show()






