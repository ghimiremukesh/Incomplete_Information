import matplotlib.pyplot as plt
import utils, icnn_pytorch_adaptive as icnn_pytorch
import torch
import numpy as np
from utils import point_dynamics
from utils import convex_hull
from itertools import product

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def point_dynamics(X, u_low, u_high, dt=0.1):
    """
    Point dynamics with velocity control in x and y direction
    :param X: State for a player
    :param u_low: lower bound for control
    :param u_high: upper bound for control
    :return: new states: [X1, X2, ...., Xn]
    """

    x = X[:, 0]
    y = X[:, 1]

    us = product([u_low, 0, u_high], repeat=2)

    us = [u_low, 0, u_high]
    X_next = []

    for ux in us:
        xdot = ux
        x_new = x + xdot * dt
        y_new = y

        X_next.append(np.concatenate((x_new.reshape(-1, 1), y_new.reshape(-1, 1)), axis=1))

    return X_next

def cav_vex(values, p, type='vex', num_ps=11):
    lower = True if type == 'vex' else False
    ps = np.linspace(0, 1, num_ps)
    values = np.vstack(values).T
    cvx_vals = np.zeros((values.shape[0], 1))
    for i in range(values.shape[0]):
        value = values[i]
        points = zip(ps, value)
        hull_points = convex_hull(points, vex=lower)
        hull_points = sorted(hull_points)
        x, y = zip(*hull_points)
        num_facets = len(hull_points) - 1
        if p[i] != 1:
            s_idx = [True if x[j] <= p[i] < x[j + 1] else False for j in range(num_facets)]
        else:
            s_idx = [True if x[j] < p[i] <= x[j + 1] else False for j in range(num_facets)]
        assert sum(s_idx) == 1, "p must belong to only one interval, check for bugs!"
        facets = np.array(list(zip(x, x[1:])))
        val_zips = np.array(list(zip(y, y[1:])))
        P = facets[s_idx].flatten()
        vals = val_zips[s_idx].flatten()
        x1, x2 = P
        y1, y2 = vals
        # calculate the value from the equation:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        cvx_vals[i] = slope * p[i] + intercept
    # p_idx = [list(ps).index(each) for each in p]
    return cvx_vals

ckpt_path = 'logs/soccer_velocity/t_1/checkpoints_dir/model_final.pth'
c_inter = 'logs/soccer_velocity/t_1/checkpoints_dir/model_final.pth'


activation = 'relu'

# initialize the model

# model = icnn_pytorch.SingleBVPNet(in_features=5, out_features=1, type='relu', mode='mlp', hidden_features=32,
#                                   num_hidden_layers=3, dropout=0)
#
# model_inter = icnn_pytorch.SingleBVPNet(in_features=5, out_features=1, type='relu', mode='mlp', hidden_features=32,
#                                   num_hidden_layers=3, dropout=0)
#
# # model = modules.SingleBVPNet(in_features=4, out_features=1, type='relu', mode='mlp', hidden_features=64,
# #                                   num_hidden_layers=2)
#
# model.to(device)
#
# model_inter.to(device)
# checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
# c_inter = torch.load(c_inter, map_location=torch.device("cpu"))
# try:
#     model_weights = checkpoint['model']
#     m_inter = c_inter['model']
# except:
#     model_weights = checkpoint
#     m_inter = c_inter
#
# model.load_state_dict(model_weights)
# model.eval()
#
# model_inter.load_state_dict(m_inter)
# model_inter.eval()

fig2 = plt.figure(figsize=(6, 6))

ax2 = fig2.gca(projection='3d')

fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))

states = [0.5]
for i in range(9):
    states.append((states[-1] - 0.005)/1.1)

t = 0.1
dt = 0.1
t_next = t - dt

# t_step = int(t * 100) // 10

ts = np.around(np.arange(dt, 1 + dt, dt), 2)
t_step = int(np.where(ts == t)[0] + 1)


# get the data
num_points = 100
x = torch.zeros((num_points, 4))
x[:, 0] = torch.linspace(-0.25, 0.25, num_points) #torch.linspace(-states[t_step-1], states[t_step-1], num_points)
x[:, 1] = 0
x[:, 2] = 0 #torch.linspace(-0.25, 0.25, num_points)
x[:, 3] = 0
p_t = torch.linspace(0, 1, num_points).reshape(-1, 1)

xo = torch.zeros((num_points*num_points, 3))
#
# xo = torch.zeros((num_points*num_points, 2))


coords = torch.cat((x, p_t), dim=1)
P = coords[:, -1].detach().cpu().numpy().squeeze()
X = coords[:, 0].detach().cpu().numpy().squeeze()
P, X = np.meshgrid(P, X)
coords_in = np.hstack((X.reshape(-1, 1), xo, P.reshape(-1, 1)))
# coords_in = np.hstack((xo, X.reshape(-1, 1), np.zeros_like(X.reshape(-1, 1)),  P.reshape(-1, 1)))

# x = torch.cat((x, p_t), dim=1)


coords_in = {'coords': torch.tensor(coords_in)}


# V_model = model(coords_in)['model_out'].detach().cpu().numpy().squeeze()
#
# V_model = V_model.reshape(P.shape)



# plot the data
# ax2.plot_surface(P, X, V_model, color='red')
# # ax2.plot_surface(P, X, V_true, color='green')
#
# tT = ax3.contour(P, X, V_model, colors='red')

ax3.set_xlabel('$p$')
ax3.set_ylabel('$X$')

# plt.clabel(tT, inline=1, fontsize=10)

plt.legend()




## ground truth for t = 0.5
#
x_prev = coords_in['coords'].numpy()
umax = 1
dmax = 1
t_next = t_next
NUM_PS = 100

xy = coords_in['coords']
u_low = -1
u_high = 1
d_low = -0.5
d_high = 0.5

g1 = utils.GOAL_1
g2 = utils.GOAL_2

G = [g1, g2]

# x_next_1 = np.vstack(point_dynamics(xy[:, :2], u_low, u_high))
# x_next_2 = np.vstack(point_dynamics(xy[:, 2:4], d_low, d_high))

x_next = utils.point_dyn_vel(x_prev, u_high, d_high)

X_next = torch.from_numpy(utils.make_pairs_vel(x_next[:, :2], x_next[:, 2:]))

vs = []


ps = np.linspace(0, 1, NUM_PS)
p = x_prev[..., -1]
for p_each in ps:
    p_next = p_each * torch.ones_like(X_next[:, 0]).reshape(-1, 1)
    X_next_p = torch.cat((X_next, p_next), dim=1)
    V_next = utils.final_cost(X_next[:, :2], X_next[:, 2:4], G, p_next.detach().numpy())

    V_next = V_next.reshape(-1, 3, 3)
    V_next = np.min(np.max(V_next, 2), 1)

    vs.append(V_next)

true_v = cav_vex(vs, p.flatten(), type='vex', num_ps=NUM_PS).reshape(1, -1, 1)

# _, true_v = np.meshgrid(p, true_v.flatten())
ax2.plot_surface(P, X, true_v.reshape(P.shape), color='green')
vT = ax3.contour(P, X, true_v.reshape(P.shape), colors='green')

# ax2.plot_surface(P, X, np.vstack(vs[0]).reshape(P.shape), color='orange')


plt.show()





