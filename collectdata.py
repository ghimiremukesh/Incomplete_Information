import numpy as np
import torch
from itertools import product
import os
import icnn_pytorch_adaptive as icnn_pytorch
import scipy.io as scio
import configargparse
from tqdm import tqdm

import utils

NUM_PS = 100
pp = configargparse.ArgumentParser()
pp.add_argument('--time', type=float, default=0.2,
                help='time-step to collect data')
opt = pp.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

activation = 'relu'


# def get_running_payoff(u, d, R1, R2, tau=0.1):
#     u = np.array(u)
#     d = np.array(d)
#     return tau * (R1 * u ** 2 - R2 * d ** 2)


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


def cav_vex(values, type='vex', num_ps=11):
    lower = True if type == 'vex' else False
    ps = np.linspace(0, 1, num_ps)
    values = np.vstack(values).T
    cvx_vals = np.zeros((values.shape[0], num_ps))
    p = np.linspace(0, 1, num_ps)
    for i in tqdm(range(values.shape[0])):
        value = values[i]
        points = zip(ps, value)
        hull_points = convex_hull(points, vex=lower)
        hull_points = sorted(hull_points)
        x, y = zip(*hull_points)
        num_facets = len(hull_points) - 1
        for k in range(len(p)):
            if p[k] != 1:
                s_idx = [True if x[j] <= p[k] < x[j + 1] else False for j in range(num_facets)]
            else:
                s_idx = [True if x[j] < p[k] <= x[j + 1] else False for j in range(num_facets)]
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
            # cvx_vals[i] = slope * p[i] + intercept
            cvx_vals[i, k] = slope * p[k] + intercept
    # p_idx = [list(ps).index(each) for each in p]
    return cvx_vals

# def make_payoff(x, y):
#     x_minus = x[1]
#     x_plus = y[1]
#     xv_minus = x[2]
#     xv_plus = y[2]
#     y_minus = x[3]
#     y_plus = y[3]
#     yv_minus = x[4]
#     yv_plus = y[4]
#
#     X_minus = (x_minus, xv_minus)
#     X_plus = (x_plus, xv_plus)
#     Y_minus = (y_minus, yv_minus)
#     Y_plus = (y_plus, yv_plus)
#
#     X = [X_minus, X_plus]
#     Y = [Y_minus, Y_plus]
#
#     pairs = np.array(list(product(X, Y)))
#
#     X_return = pairs.reshape(-1, 4)
#     X_return = np.hstack((x[0] * np.ones((4, 1)), X_return))
#
#     return X_return

def dynamics(x, u, d, DT):
    def get_p2_a(x, d_max):
        v = x[..., 4]  # velocity
        # a_l = -d_max * (1 - np.maximum(np.zeros_like(v), 0.5 * np.sign(v)))
        # a_r = d_max * (1 - np.maximum(np.zeros_like(v), 0.5 * np.sign(-v)))
        a_l = -d_max * np.ones_like(v)
        a_r = d_max * np.ones_like(v)

        return a_l, a_r

    # U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
    U = np.array([-np.ones_like(x[..., 2]).reshape(-1, 1), np.zeros_like(x[..., 2]).reshape(-1, 1),
                  np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
    d1, d2 = get_p2_a(x, d)
    D = np.vstack((d1, np.zeros_like(d1), d2)).T   # add zero to action
    dt = DT
    x1_new = x[..., 1].reshape(-1, 1) + x[..., 2].reshape(-1, 1) * dt + 0.5 * U * dt ** 2
    x2_new = x[..., 3].reshape(-1, 1) + x[..., 4].reshape(-1, 1) * dt + 0.5 * D * dt ** 2
    v1_new = x[..., 2].reshape(-1, 1) + U * dt
    v2_new = x[..., 4].reshape(-1, 1) + D * dt

    # v_new = np.array([x[..., 2] + a for a in da_v]).T.reshape(-1, 1)
    # p_new = np.multiply(x[:, 3].reshape(-1, 1),
    #                     np.ones((x.shape[0], 4))).reshape(-1, 1)
    X_new = np.hstack((x1_new.reshape(-1, 1), v1_new.reshape(-1, 1), x2_new.reshape(-1, 1),
                       v2_new.reshape(-1, 1)))  # returns new states, n x 8

    return X_new
## for three action choices

def make_payoff(x, y, z):
    x_minus = x[1]
    x_zero = y[1]
    x_plus = z[1]
    xv_minus = x[2]
    xv_zero = y[2]
    xv_plus = z[2]
    y_minus = x[3]
    y_zero = y[3]
    y_plus = z[3]
    yv_minus = x[4]
    yv_zero = y[4]
    yv_plus = z[4]

    X_minus = (x_minus, xv_minus)
    X_zero = (x_zero, xv_zero)
    X_plus = (x_plus, xv_plus)
    Y_minus = (y_minus, yv_minus)
    Y_zero = (y_zero, yv_zero)
    Y_plus = (y_plus, yv_plus)

    X = [X_minus, X_zero, X_plus]
    Y = [Y_minus, Y_zero, Y_plus]

    pairs = np.array(list(product(X, Y)))

    X_return = pairs.reshape(-1, 4)
    X_return = np.hstack((x[0] * np.ones((9, 1)), X_return))

    return X_return

## for three action choices
# def dynamics(x, u, d, DT):
#     def get_p2_a(x, d_max):
#         v = x[..., 4]  # velocity
#
#         a_l = -d_max * np.ones_like(v)
#         a_r = d_max * np.ones_like(v)
#
#         # a_l[np.where(v < 0)] = -d_max
#         # a_r[np.where(v < 0)] = 0.5 * d_max
#         #
#         # a_l[np.where(v > 0)] = -0.5 * d_max
#         # a_r[np.where(v > 0)] = d_max
#
#         return a_l, a_r
#
#     U = np.array([-u * np.ones_like(x[..., 2]).reshape(-1, 1), np.zeros_like(x[..., 2]).reshape(-1, 1),
#                   u * np.ones_like(x[..., 2]).reshape(-1, 1)]).T.squeeze()
#     d1, d2 = get_p2_a(x, d)
#     D = np.vstack((d1, np.zeros_like(d1), d2)).T
#     dt = DT
#     x1_new = x[..., 1].reshape(-1, 1) + x[..., 2].reshape(-1, 1) * dt + 0.5 * U * dt ** 2
#     x2_new = x[..., 3].reshape(-1, 1) + x[..., 4].reshape(-1, 1) * dt + 0.5 * D * dt ** 2
#     v1_new = x[..., 2].reshape(-1, 1) + U * dt
#     v2_new = x[..., 4].reshape(-1, 1) + D * dt
#
#     X_new = np.hstack((x1_new.reshape(-1, 1), v1_new.reshape(-1, 1), x2_new.reshape(-1, 1),
#                        v2_new.reshape(-1, 1)))  # returns new states, n x 8
#
#     return X_new


if __name__ == "__main__":
    num_points = 1000
    extra_points = 1000  # sample around 0
    t = opt.time
    dt = 0.05
    # t_step = int(t * 100) // 10
    ts = np.around(np.arange(dt, 1+dt, dt), 2)
    t_step = int(np.where(ts == t)[0] + 1)
    # t_step = int(np.ceil((t * 10)))
    umax = 1
    dmax = 1
    u_map = {-1: 1, 0: 0, 1: 1}
    d_map = {-1: -1, 0: 0, 1: 1}
    R1 = 1
    R2 = 5
    num_ps = 100
    logging_root = 'logs/'
    save_root = f'hexner_test_case_R2_timestep_{dt}/'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    up = 0.5
    states = [up]
    for _ in range(19):  # change to 9 for 10 timesteps
        states.append((states[-1] - 0.005) / 1.1)

    # up = (1 - t) * 0.5
    x = torch.zeros(num_points, 4).uniform_(-states[t_step - 1], states[t_step - 1])  # x1, v1, x2, v2
    # x_extra_1 = torch.zeros(extra_points, 4).uniform_(-0.1, 0.1)
    x_extra_2 = torch.zeros(extra_points, 2).uniform_(-0.1, 0.1)
    x_extra_2 = torch.cat((x_extra_2, x_extra_2), dim=1)
    x_extra_3 = torch.zeros(extra_points, 2).uniform_(-0.05, 0.05)
    x_extra_3 = torch.cat((x_extra_3, x_extra_3), dim=1)
    x_extra_4 = torch.zeros(extra_points, 2).uniform_(-0.01, 0.01)
    x_extra_4 = torch.cat((x_extra_4, x_extra_4), dim=1)

    x1_same = torch.zeros(num_points, 2).uniform_(-states[t_step - 1], states[t_step - 1])
    x_same = torch.cat((x1_same, x1_same), dim=1)


    x = torch.cat((x, x_same, x_extra_2, x_extra_3, x_extra_4), dim=0)


    time = torch.ones(x.shape[0], 1) * t

    coords = torch.cat((time, x), dim=1)

    x_prev = coords.detach().cpu().numpy()
    if t == dt:
        t_next = t - dt
        x_next = torch.from_numpy(dynamics(x_prev, umax, dmax, DT=dt))
        x_next = torch.cat((t_next * torch.ones((x_next.shape[0], 1)), x_next), dim=1)
        # x_next = np.array([make_payoff(x_next[i, :], x_next[i + 1, :])
        #                    for i in range(0, len(x_next), 2)]).reshape(-1, 5)  # change this to vecotrized op.
        x_next = np.array([make_payoff(x_next[i, :], x_next[i + 1, :], x_next[i + 2, :])
                           for i in range(0, len(x_next), 3)]).reshape(-1, 5)
        # coords_in = {'coords_in': torch.tensor(x_next).to(device)}
        x_next = torch.from_numpy(x_next)
        vs = []
        ps = np.linspace(0, 1, NUM_PS)
        # p = x_prev[..., -1]
        for p_each in ps:
            p_next = p_each * torch.ones_like(x_next[:, 1]).reshape(-1, 1)
            x_next_p = torch.cat((x_next, p_next), dim=1)
            coords_in = {'coords': x_next_p.to(device)}
            # v_next = val_model(coords_in)['model_out'].detach().cpu().numpy()
            v_next = utils.final_value(x_next_p[:, 1], x_next_p[:, 2],  x_next_p[:, 3],
                                       x_next_p[:, 4], x_next_p[:, -1]).numpy()
            # v_next = v_next.reshape(-1, 2, 2) + dt * utils.get_running_payoff(
            #     list(u_map.values()), list(d_map.values()), R1, R2, tau=dt).reshape(-1, 2, 2)
            v_next = v_next.reshape(-1, 3, 3) + dt * utils.get_running_payoff(
                list(u_map.values()), list(d_map.values()), R1, R2, tau=dt).reshape(-1, 3, 3)
            v_next = np.min(np.max(v_next, 2), 1)
            vs.append(v_next)

        # true_v = cav_vex(vs, p.flatten(), type='vex', num_ps=NUM_PS).reshape(1, -1, 1)
        true_v = cav_vex(vs, type='vex', num_ps=NUM_PS).reshape(1, -1, 1)

        ps = torch.linspace(0, 1, 100)
        p = ps.repeat([len(x), 1]).reshape(-1, 1)
        x = torch.vstack([x[i].repeat([NUM_PS, 1]) for i in range(len(x))])
        coords = torch.cat((x, p), dim=1)

        time = torch.ones(x.shape[0], 1) * t
        coords = torch.cat((time, coords), dim=1)

        x_prev = coords.detach().cpu().numpy()

        gt = {'states': np.vstack(x_prev),
              'values': np.vstack(true_v)}

        scio.savemat(os.path.join(save_root, f'train_data_t_{t:.2f}.mat'), gt)

    else:
        t_next = t - dt

        load_dir = os.path.join(logging_root, f'hexner_train_R2_timestep_{dt}/t_{t_step - 1}/')
        start_epoch = 4  # num_epoch - 1 from training

        val_model = icnn_pytorch.SingleBVPNet(in_features=6, out_features=1, type=activation, mode='mlp',
                                              hidden_features=128, num_hidden_layers=2, dropout=0)
        val_model.to(device)
        model_path = os.path.join(load_dir, 'checkpoints_dir', 'model_epoch_%04d.pth' % start_epoch)
        checkpoint = torch.load(model_path, map_location=device)
        val_model.load_state_dict(checkpoint['model'])
        val_model.eval()

        x_next = torch.from_numpy(dynamics(x_prev, umax, dmax, DT=dt))
        x_next = torch.cat((t_next * torch.ones((x_next.shape[0], 1)), x_next), dim=1)
        # x_next = np.array([make_payoff(x_next[i, :], x_next[i + 1, :])
        #                    for i in range(0, len(x_next), 2)]).reshape(-1, 5)  # change this to vecotrized op.
        x_next = np.array([make_payoff(x_next[i, :], x_next[i + 1, :], x_next[i + 2, :])
                           for i in range(0, len(x_next), 3)]).reshape(-1, 5)
        # coords_in = {'coords_in': torch.tensor(x_next).to(device)}
        x_next = torch.from_numpy(x_next.astype(np.float32))
        vs = []
        ps = np.linspace(0, 1, NUM_PS)
        for p_each in ps:
            p_next = p_each * torch.ones_like(x_next[:, 1]).reshape(-1, 1)
            x_next_p = torch.cat((x_next, p_next), dim=1)
            coords_in = {'coords': x_next_p.to(device)}
            v_next = val_model(coords_in)['model_out'].detach().cpu().numpy()
            # v_next = v_next.reshape(-1, 2, 2) + dt * utils.get_running_payoff(
            #     list(u_map.values()), list(d_map.values()), R1, R2, tau=dt).reshape(-1, 2, 2)
            v_next = v_next.reshape(-1, 3, 3) + dt * utils.get_running_payoff(
                list(u_map.values()), list(d_map.values()), R1, R2, tau=dt).reshape(-1, 3, 3)
            v_next = np.min(np.max(v_next, 2), 1)
            vs.append(v_next)

        true_v = cav_vex(vs, type='vex', num_ps=NUM_PS).reshape(1, -1, 1)

        ps = torch.linspace(0, 1, 100)
        p = ps.repeat([len(x), 1]).reshape(-1, 1)
        x = torch.vstack([x[i].repeat([NUM_PS, 1]) for i in range(len(x))])
        coords = torch.cat((x, p), dim=1)

        time = torch.ones(x.shape[0], 1) * t
        coords = torch.cat((time, coords), dim=1)

        x_prev = coords.detach().cpu().numpy()

        gt = {'states': np.vstack(x_prev),
              'values': np.vstack(true_v)}

        scio.savemat(os.path.join(save_root, f'train_data_t_{t:.2f}.mat'), gt)
