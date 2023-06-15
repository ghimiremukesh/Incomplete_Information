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


## Start here


ckpt_path = '../train_new_case_follow_zero_relative/t_9/checkpoints_dir/model_final.pth'  # current model
c_inter = '../train_new_case_follow_zero_relative/t_8/checkpoints_dir/model_final.pth'  # model for next time-step

activation = 'relu'

# initialize the model

model = icnn_pytorch.SingleBVPNet(in_features=5, out_features=1, type=activation, mode='mlp', hidden_features=128,
                                  num_hidden_layers=2)

model_inter = icnn_pytorch.SingleBVPNet(in_features=5, out_features=1, type=activation, mode='mlp', hidden_features=128,
                                        num_hidden_layers=2)

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

# state-space becomes narrower as we go back in time
states = [0.5]
for i in range(9):
    states.append((states[-1] - 0.01) / 1.1)

t = 0.9  # current time-step (backwards)
t_next = t - 0.1  # next time-step

t_step = int(t * 100) // 10

# get the data
num_points = 10
x = torch.zeros((num_points, 4))
x[:, 0] = t  # time
x[:, 1] = torch.linspace(-states[t_step - 1], states[t_step - 1], num_points)  # position
x[:, 2] = 0.  # p1's velocity
x[:, 3] = 0.  # p2's velocity
p_t = torch.linspace(0, 1, num_points).reshape(-1, 1)  # belief

t = t * torch.ones((num_points * num_points, 1))
x_o = torch.zeros((num_points * num_points, 2))

coords = torch.cat((x, p_t), dim=1)
P = coords[:, -1].detach().cpu().numpy().squeeze()
X = coords[:, 1:-1].detach().cpu().numpy().squeeze()  # remove time
P, X = np.meshgrid(P, X[:, 0])  # create mesh between belief and position
coords_in = np.hstack((t, X.reshape(-1, 1), x_o, P.reshape(-1, 1)[:t.shape[0]]))
x = torch.cat((x, p_t), dim=1)

coords_in = {'coords': torch.tensor(coords_in)}

V_model = model(coords_in)['model_out'].detach().cpu().numpy().squeeze()

V_model = V_model.reshape(P.shape)

# plot the data

ax2.plot_surface(P, X, V_model, color='red')  # model value at some time t

tT = ax3.contour(P, X, V_model, colors='red')

ax3.set_xlabel('$p$')
ax3.set_ylabel('$V$')

#  now plot for time t+dt

x_prev = coords_in['coords'].numpy()
umax = 1
dmax = 1
t_next = t_next
NUM_PS = 100

x_next = torch.from_numpy(utils.dynamics_zero_relative_high(x_prev, umax, dmax, DT=0.1))  # computes the next states
x_next = torch.cat((t_next * torch.ones((x_next.shape[0], 1)), x_next), dim=1)

vs = []  # store values for each state at all ps
ps = np.linspace(0, 1, NUM_PS)
p = x_prev[..., -1]
for p_each in ps:
    p_next = p_each * torch.ones_like(x_next[:, 1]).reshape(-1, 1)
    x_next_p = torch.cat((x_next, p_next), dim=1)
    coords_in = {'coords': x_next_p.to(device)}
    # v_next = utils.final_value_minmax(x_next_p[:, 1], x_next_p[:, -1]).numpy()  #  uncomment this to calculate value at T-dt and comment below
    v_next = model_inter(coords_in)['model_out'].detach().cpu().numpy()
    v_next = v_next.reshape(-1, 3, 3)
    v_next = np.min(np.max(v_next, 2), 1)
    vs.append(v_next)

true_v = cav_vex(vs, p.flatten(), type='vex', num_ps=NUM_PS).reshape(1, -1, 1)  # compute convexification of values

ax2.plot_surface(P, X, true_v.reshape(P.shape), color='green')
vT = ax3.contour(P, X, true_v.reshape(P.shape), colors='green')

plt.show()
