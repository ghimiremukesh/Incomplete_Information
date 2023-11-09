import matplotlib.pyplot as plt
import utils, icnn_pytorch_adaptive as icnn_pytorch
import torch
import numpy as np
from utils import point_dynamics
from utils import convex_hull
from itertools import product

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def point_dynamics(x, u_max, d_max, dt=0.1):
    """
    Point dynamics with acceleration control for all possible actions
    :param X: Joint state of players
    :param u_max: upper bound for control
    :param d_max: upper bound for control
    :return: new states: [X1, X2, ...., Xn]
    """

    # get a map of actions u_map and d_map
    us = list(product([-u_max, 0, u_max], repeat=2))
    ds = list(product([-d_max, 0, d_max], repeat=2))
    umap = {k: v for (k, v) in enumerate(us)}
    dmap = {k: v for (k, v) in enumerate(ds)}

    U = np.array([i for i in range(9)]).reshape(1, -1)
    U = np.repeat(U, x[..., 2].shape[0], axis=0)

    D = np.array([i for i in range(9)]).reshape(1, -1)
    D = np.repeat(D, x[..., 2].shape[0], axis=0)

    action_array_u = np.array([umap[i] for i in range(len(umap))])[U]
    action_array_d = np.array([dmap[i] for i in range(len(dmap))])[D]

    x1 = x[:, 0].reshape(-1, 1)
    y1 = x[:, 1].reshape(-1, 1)
    vx1 = x[:, 2].reshape(-1, 1)
    vy1 = x[:, 3].reshape(-1, 1)

    x2 = x[:, 4].reshape(-1, 1)
    y2 = x[:, 5].reshape(-1, 1)
    vx2 = x[:, 6].reshape(-1, 1)
    vy2 = x[:, 7].reshape(-1, 1)

    x1dot = vx1
    y1dot = vy1

    x2dot = vx2
    y2dot = vy2

    vx1dot = action_array_u[:, :, 0]
    vy1dot = action_array_u[:, :, 1]

    vx2dot = action_array_d[:, :, 0]
    vy2dot = action_array_d[:, :, 1]

    x1_new = x1 + x1dot * dt + 0.5 * vx1dot * (dt ** 2)
    y1_new = y1 + y1dot * dt + 0.5 * vy1dot * (dt ** 2)
    vx1_new = vx1 + vx1dot * dt
    vy1_new = vy1 + vy1dot * dt

    x2_new = x2 + x2dot * dt + 0.5 * vx2dot * (dt ** 2)
    y2_new = y2 + y2dot * dt + 0.5 * vy2dot * (dt ** 2)
    vx2_new = vx2 + vx2dot * dt
    vy2_new = vy2 + vy2dot * dt

    X_new = np.hstack((x1_new.reshape(-1, 1), y1_new.reshape(-1, 1), vx1_new.reshape(-1, 1), vy1_new.reshape(-1, 1),
                       x2_new.reshape(-1, 1), y2_new.reshape(-1, 1), vx2_new.reshape(-1, 1), vy2_new.reshape(-1, 1)))

    return X_new

def point_dynamics_test(x, u_max, d_max, dt=0.1):
    """
    Point dynamics with acceleration control for all possible actions
    :param X: Joint state of players
    :param u_max: upper bound for control
    :param d_max: upper bound for control
    :return: new states: [X1, X2, ...., Xn]
    """

    # get a map of actions u_map and d_map
    us = list(product([-u_max, 0, u_max], repeat=2))
    ds = list(product([-d_max, 0, d_max], repeat=2))
    umap = {k: v for (k, v) in enumerate(us)}
    dmap = {k: v for (k, v) in enumerate(ds)}

    U = np.array([i for i in range(9)]).reshape(1, -1)
    U = np.repeat(U, x[..., 2].shape[0], axis=0)

    D = np.array([i for i in range(9)]).reshape(1, -1)
    D = np.repeat(D, x[..., 2].shape[0], axis=0)

    action_array_u = np.array([umap[i] for i in range(len(umap))])[U]
    action_array_d = np.array([dmap[i] for i in range(len(dmap))])[D]

    x1 = x[:, 0].reshape(-1, 1)
    y1 = x[:, 1].reshape(-1, 1)
    vx1 = x[:, 2].reshape(-1, 1)
    vy1 = x[:, 3].reshape(-1, 1)

    x2 = x[:, 4].reshape(-1, 1)
    y2 = x[:, 5].reshape(-1, 1)
    vx2 = x[:, 6].reshape(-1, 1)
    vy2 = x[:, 7].reshape(-1, 1)

    x1dot = vx1
    y1dot = vy1

    x2dot = vx2
    y2dot = vy2

    vx1dot = 0.9 * action_array_u[:, :, 0]
    vy1dot = 0.9 * action_array_u[:, :, 1]

    vx2dot = action_array_d[:, :, 0]
    vy2dot = action_array_d[:, :, 1]

    x1_new = x1 + x1dot * dt + 0.5 * vx1dot * (dt ** 2)
    y1_new = y1 + y1dot * dt + 0.5 * vy1dot * (dt ** 2)
    vx1_new = vx1 + vx1dot * dt
    vy1_new = vy1 + vy1dot * dt

    x2_new = x2 + x2dot * dt + 0.5 * vx2dot * (dt ** 2)
    y2_new = y2 + y2dot * dt + 0.5 * vy2dot * (dt ** 2)
    vx2_new = vx2 + vx2dot * dt
    vy2_new = vy2 + vy2dot * dt

    X_new = np.hstack((x1_new.reshape(-1, 1), y1_new.reshape(-1, 1), vx1_new.reshape(-1, 1), vy1_new.reshape(-1, 1),
                       x2_new.reshape(-1, 1), y2_new.reshape(-1, 1), vx2_new.reshape(-1, 1), vy2_new.reshape(-1, 1)))

    return X_new

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

ckpt_path = 'logs/soccer_constrained/t_1/checkpoints_dir/model_final.pth'
c_inter = 'logs/soccer_constrained/t_1/checkpoints_dir/model_final.pth'

ckpt_path = 'logs/soccer_uncons_effort_square/t_1/checkpoints_dir/model_final.pth'
c_inter = 'logs/soccer_uncons_effort_square/t_1/checkpoints_dir/model_final.pth'

# ckpt_path = 'logs/soccer_constrained/from_cluster/relu/model_final.pth'
# c_inter = 'logs/soccer_constrained/from_cluster/relu/model_final.pth'

# ckpt_path = 'logs/soccer/t_1/checkpoints_dir/model_final.pth'
# c_inter = 'logs/soccer/t_1/checkpoints_dir/model_final.pth'


activation = 'relu'

game = 'uncons'

# initialize the model
 # for cluster (deep+wide network)

model = icnn_pytorch.SingleBVPNet(in_features=9, out_features=1, type='relu', mode='mlp', hidden_features=72,
                                  num_hidden_layers=5, dropout=0)

model_inter = icnn_pytorch.SingleBVPNet(in_features=9, out_features=1, type='relu', mode='mlp', hidden_features=72,
                                  num_hidden_layers=5, dropout=0)

# (shallow + skinny network)

# model = icnn_pytorch.SingleBVPNet(in_features=9, out_features=1, type='relu', mode='mlp', hidden_features=72,
#                                   num_hidden_layers=5, dropout=0)
#
# model_inter = icnn_pytorch.SingleBVPNet(in_features=9, out_features=1, type='relu', mode='mlp', hidden_features=72,
#                                   num_hidden_layers=5, dropout=0)

# model = modules.SingleBVPNet(in_features=4, out_features=1, type='relu', mode='mlp', hidden_features=64,
#                                   num_hidden_layers=2)

model.to(device)

model_inter.to(device)


checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
c_inter = torch.load(c_inter, map_location=torch.device("cpu"))

try:
    model_weights = checkpoint['model']
    m_inter = c_inter['model']
except:
    model_weights = checkpoint
    m_inter = c_inter

model.load_state_dict(model_weights)
model.eval()

model_inter.load_state_dict(m_inter)
model_inter.eval()

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
num_points = 50
x = torch.zeros((num_points, 8))
x[:, 0] = torch.linspace(-1, 1, num_points) #torch.linspace(-states[t_step-1], states[t_step-1], num_points)
x[:, 1] = 0 #torch.linspace(-1, 1, num_points)
x[:, 4] = 0 #torch.zeros(num_points, ).uniform_(-1, 1)
x[:, 5] = 0
p_t = torch.linspace(0, 1, num_points).reshape(-1, 1)

xo = torch.zeros((num_points*num_points, 7))

# xo[:, 3] = -1
#
# xl = torch.zeros((num_points*num_points, 4))

# xo[:, 4] = torch.linspace(-1, 1, num_points*num_points)
#
# xo1 = torch.zeros((num_points*num_points, 1))
# xo2 = torch.zeros((num_points*num_points, 2))

coords = torch.cat((x, p_t), dim=1)
P = coords[:, -1].detach().cpu().numpy().squeeze()
X = coords[:, 0].detach().cpu().numpy().squeeze()
# X = coords[:, 1].detach().cpu().numpy().squeeze()
P, X = np.meshgrid(P, X)
coords_in = np.hstack((X.reshape(-1, 1), xo, P.reshape(-1, 1)))
# coords_in = np.hstack((xo1, X.reshape(-1, 1), xo2, xl, P.reshape(-1, 1)))

# x = torch.cat((x, p_t), dim=1)

coords_in = {'coords': torch.tensor(coords_in)}


V_model = model(coords_in)['model_out'].detach().cpu().numpy().squeeze()
# #
V_model = V_model.reshape(P.shape)



# plot the data
ax2.plot_surface(P, X, V_model, color='red')
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
NUM_PS = num_points

u_low = -2
u_high = 2
d_low = -1
d_high = 1

g1 = utils.GOAL_1
g2 = utils.GOAL_2

G = [g1, g2]

# x_next_1 = np.vstack(point_dynamics(xy[:, :4], u_low, u_high))
# x_next_2 = np.vstack(point_dynamics(xy[:, 4:-1], d_low, d_high))

X_next = point_dynamics(x_prev, u_high, d_high)

X_next = torch.from_numpy(utils.make_pairs(X_next[:, :4], X_next[:, 4:8]))

vs = []


ps = np.linspace(0, 1, NUM_PS)
p = x_prev[..., -1]
for p_each in ps:
    p_next = p_each * torch.ones_like(X_next[:, 0]).reshape(-1, 1)
    X_next_p = torch.cat((X_next, p_next), dim=1)
    V_next = utils.final_cost(X_next[:, :2], X_next[:, 4:6], G, p_next.detach().numpy(), game=game)
    # V_next = model_inter({'coords': X_next_p.to(torch.float32)})['model_out'].detach().cpu().numpy().squeeze()
    V_next = V_next.reshape(-1, 9, 9) + dt * utils.inst_cost(u_high, d_high, 1, 1).reshape(-1, 9, 9)
    V_next = np.min(np.max(V_next, 2), 1)

    vs.append(V_next)

true_v = cav_vex(vs, p.flatten(), type='vex', num_ps=NUM_PS).reshape(1, -1, 1)

# _, true_v = np.meshgrid(p, true_v.flatten())
ax2.plot_surface(P, X, true_v.reshape(P.shape), color='green')
vT = ax3.contour(P, X, true_v.reshape(P.shape), colors='green')

vals = np.vstack(vs).T
minimax_v = np.array(([vals[i, i % num_points] for i in range(num_points * num_points)]))
#
# ax2.plot_surface(P, X, minimax_v.reshape(P.shape), color='orange')






plt.show()





