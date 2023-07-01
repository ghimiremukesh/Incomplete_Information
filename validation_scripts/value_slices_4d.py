import matplotlib.pyplot as plt
import utils, icnn_pytorch_adaptive as icnn_pytorch
import torch
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def convex_hull(points, vex=True):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """
    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    if vex:
        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
    else:
        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower[:] if vex else upper[:]

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

# def dynamics(x, u, d):
#     u = [-u, u]
#     d = [-d, d]
#     dt = 0.5
#     da = np.array([a - b for a, b in product(u, d)])
#     da_v = da * dt
#     da_x = da * 0.5 * dt ** 2
#     x_new = np.array([x[..., 1] + dt * x[..., 2] + a for a in da_x]).T.reshape(-1, 1)
#     v_new = np.array([x[..., 2] + a for a in da_v]).T.reshape(-1, 1)
#     # p_new = np.multiply(x[:, 3].reshape(-1, 1),
#     #                     np.ones((x.shape[0], 4))).reshape(-1, 1)
#     X_new = np.hstack((x_new, v_new))
#
#     return X_new


ckpt_path = '../logs/hexner_train_R2_timestep_0.2/t_1/checkpoints_dir/model_final.pth'
c_inter = '../logs/hexner_train_R2_timestep_0.2/t_1/checkpoints_dir/model_final.pth'


activation = 'relu'

# initialize the model

model = icnn_pytorch.SingleBVPNet(in_features=6, out_features=1, type='relu', mode='mlp', hidden_features=128,
                                  num_hidden_layers=2, dropout=0)

model_inter = icnn_pytorch.SingleBVPNet(in_features=6, out_features=1, type='relu', mode='mlp', hidden_features=128,
                                  num_hidden_layers=2, dropout=0)

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

t = 0.2
dt = 0.2
t_next = t - dt

# t_step = int(t * 100) // 10

ts = np.around(np.arange(dt, 1 + dt, dt), 2)
t_step = int(np.where(ts == t)[0] + 1)

u_map = {0: -1, 1: 1}
d_map = {0: -1, 1: 1}
R1 = 1
R2 = 5

# get the data
num_points = 10
x = torch.zeros((num_points, 5))
x[:, 0] = t
x[:, 1] = torch.linspace(-states[t_step-1], states[t_step-1], num_points)
x[:, 2] = 0
p_t = torch.linspace(0, 1, num_points).reshape(-1, 1)

t = t*torch.ones((num_points*num_points, 1))
v = 0*torch.ones((num_points*num_points, 1))
x_o = torch.zeros((num_points*num_points, 3))


coords = torch.cat((x, p_t), dim=1)
P = coords[:, -1].detach().cpu().numpy().squeeze()
X = coords[:, 1:5].detach().cpu().numpy().squeeze()
P, X = np.meshgrid(P, X[:, 0])
coords_in = np.hstack((t, X.reshape(-1, 1), x_o, P.reshape(-1, 1)[:t.shape[0]]))
x = torch.cat((x, p_t), dim=1)

coords_in = {'coords': torch.tensor(coords_in)}


V_model = model(coords_in)['model_out'].detach().cpu().numpy().squeeze()

V_model = V_model.reshape(P.shape)



# plot the data
ax2.plot_surface(P, X, V_model, color='red')
# ax2.plot_surface(P, X, V_true, color='green')

tT = ax3.contour(P, X, V_model, colors='red')

ax3.set_xlabel('$p$')
ax3.set_ylabel('$V$')

# plt.clabel(tT, inline=1, fontsize=10)

plt.legend()




## ground truth for t = 0.5
#
x_prev = coords_in['coords'].numpy()
umax = 1
dmax = 1
t_next = t_next
NUM_PS = 100

# x_next = torch.from_numpy(utils.dynamics_4d(x_prev, umax, dmax, DT=0.1))
x_next = torch.from_numpy(utils.dynamics(x_prev, umax, dmax, DT=0.1))
x_next = torch.cat((t_next * torch.ones((x_next.shape[0], 1)), x_next), dim=1)
x_next = np.array([utils.make_payoff(x_next[i, :], x_next[i + 1, :])
                   for i in range(0, len(x_next), 2)]).reshape(-1, 5)  # change this to vecotrized op.
# coords_in = {'coords_in': torch.tensor(x_next).to(device)}
x_next = torch.from_numpy(x_next.astype(np.float32))
vs = []
ps = np.linspace(0, 1, NUM_PS)
p = x_prev[..., -1]
for p_each in ps:
    p_next = p_each * torch.ones_like(x_next[:, 1]).reshape(-1, 1)
    x_next_p = torch.cat((x_next, p_next), dim=1)
    coords_in = {'coords': x_next_p.to(device)}
    # v_next = utils.final_value_minmax((x_next_p[:, 1] - x_next_p[:, 3]), x_next_p[:, -1]).numpy()
    v_next = utils.final_value(x_next_p[:, 1], x_next_p[:, 2], x_next_p[:, 3], x_next_p[:, 4], x_next_p[:, -1]).numpy()

    # v_next = model_inter(coords_in)['model_out'].detach().cpu().numpy()

    v_next = v_next.reshape(-1, 2, 2) + dt * utils.get_running_payoff(
        list(u_map.values()), list(d_map.values()), R1, R2, tau=dt).reshape(-1, 2, 2)

    # u_idxs = np.argmin(np.max(v_next, 2), 1)
    # d_idxs = np.argmax(np.min(v_next, 1), 1)
    # u = [u_map[idx] for idx in u_idxs]
    # d = [d_map[idx] for idx in d_idxs]
    v_next = np.min(np.max(v_next, 2), 1)
    vs.append(v_next)

true_v = cav_vex(vs, p.flatten(), type='vex', num_ps=NUM_PS).reshape(1, -1, 1)

# _, true_v = np.meshgrid(p, true_v.flatten())
ax2.plot_surface(P, X, true_v.reshape(P.shape), color='green')
vT = ax3.contour(P, X, true_v.reshape(P.shape), colors='green')

plt.show()





