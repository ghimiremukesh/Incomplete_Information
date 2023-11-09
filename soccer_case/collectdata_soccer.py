import multiprocessing
import sys

sys.path.append('../')
import numpy as np
import torch
import icnn_pytorch_adaptive as icnn
import configargparse
import os
from soccer_case import utils
from itertools import product
from tqdm import tqdm
import scipy.io as scio
from utils import convex_hull

import matplotlib.pyplot as plt


# Data Collection for Training

NUM_PS = 100
pp = configargparse.ArgumentParser()
pp.add_argument('--time', type=float, default=0.1,
                help='time-step to collect data')
opt = pp.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

activation = 'relu'

Q = multiprocessing.Queue()
def process_p(p, X_next, G):
    p = p * np.ones_like(X_next[:, 0]).reshape(-1, 1)
    V_next = utils.final_cost(X_next[:, :2], X_next[:, 4:6], G, p)
    V_next = V_next.reshape(-1, 9, 9)
    V_next = np.min(np.max(V_next, 2), 1)
    Q.put(V_next)
    return V_next

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


def point_dynamics(x, u_max, d_max, dt=0.1):
    """
    Point dynamics with acceleration control for all possible actions
    :param X: Joint state of players
    :param u_max: upper bound for control
    :param d_max: upper bound for control
    :return: new states: [X1, X2, ...., Xn] containing all possible states
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

def point_dynamics_velocity(X, u_low, u_high, dt=0.1):
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
    X_next = []

    for ux, uy in us:
        xdot = ux
        ydot = uy
        x_new = x + xdot * dt
        y_new = y + ydot * dt

        X_next.append(np.concatenate((x_new.reshape(-1, 1), y_new.reshape(-1, 1)), axis=1))

    return X_next

if __name__ == '__main__':
    model = 'particle'  # 'unicycle'
    num_points = 5000
    num_players = 2
    num_states = 4  # x, y, vx, vy for each player


    extra_points = 5000  # sample around unsafe region

    game = 'cons'
    u_low = -3
    u_high = 3
    d_low = -3
    d_high = 3

    g1 = utils.GOAL_1
    g2 = utils.GOAL_2

    R1 = 0.01
    R2 = 0.01


    G = [g1, g2]

    t = opt.time
    dt = 0.1

    ts = np.around(np.arange(dt, 1 + dt, dt), 2)
    t_step = int(np.where(ts == t)[0] + 1)

    NUM_PS = 100

    logging_root = 'logs/'
    save_root = f'soccer_uncons_effort_square/'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    xy = torch.zeros(num_points, num_states * num_players).uniform_(-1, 1)
    xy_extra = torch.zeros(num_points, num_states * num_players).uniform_(-0.5, 0.5)
    xy = torch.cat((xy, xy_extra), dim=0)

    # xy[:, 0] = torch.linspace(-1, 1, num_points)
    # xy[:, 1:] = 0

    # xy_unsafe = torch.zeros(extra_points, num_states * num_players).uniform_(-0.01, 0.01)
    # xy_unsafe2 = torch.zeros(extra_points, num_states * num_players).uniform_(-0.02, 0.02)
    # xy_unsafe3 = torch.zeros(extra_points, num_states * num_players).uniform_(-0.03, 0.03)
    # xy_unsafe4 = torch.zeros(extra_points, num_states * num_players).uniform_(-0.04, 0.04)
    # xy_unsafe5 = torch.zeros(extra_points, num_states * num_players).uniform_(-0.05, 0.05)


    # xy = torch.cat((xy, xy_unsafe, xy_unsafe2, xy_unsafe3, xy_unsafe4, xy_unsafe5), dim=0)
    # xy = torch.cat((xy, xy_unsafe, xy_unsafe5), dim=0)

    time = torch.ones(xy.shape[0], 1) * t

    if t == dt:
        t_next = t - dt
        # x_next_1 = np.vstack(point_dynamics(xy[:, :num_states], u_low, u_high))
        # x_next_2 = np.vstack(point_dynamics(xy[:, num_states:], d_low, d_high))
        x_next = point_dynamics(xy, u_high, d_high)

        X_next = torch.from_numpy(utils.make_pairs(x_next[:, :4], x_next[:, 4:8]))

        vs = []
        ps = np.linspace(0, 1, NUM_PS)

        for p_each in tqdm(ps):
            p = p_each * torch.ones_like(X_next[:, 0]).reshape(-1, 1)
            X_next_p = torch.cat((X_next, p), dim=1)
            V_next = utils.final_cost(X_next[:, :2], X_next[:, 4:6], G, p.detach().numpy(), game=game)
            # V_next = list(map(utils.final_cost, X_next[:, :2], X_next[:, 2:4],
            #                   [G for _ in range(X_next_p.shape[0])], p.detach().numpy()))

            V_next = V_next.reshape(-1, 9, 9) + dt * utils.inst_cost(u_high, d_high, R1, R2).reshape(-1, 9, 9)
            V_next = np.min(np.max(V_next, 2), 1)

            vs.append(V_next)

        true_v = cav_vex(vs, type='vex', num_ps=NUM_PS).reshape(1, -1, 1)


        ps = torch.linspace(0, 1, 100)
        p = ps.repeat([len(xy), 1]).reshape(-1, 1)
        x = torch.vstack([xy[i].repeat([NUM_PS, 1]) for i in range(len(xy))])
        coords = torch.cat((x, p), dim=1)

        # time = torch.ones(x.shape[0], 1) * t
        # coords = torch.cat((time, coords), dim=1)

        x_prev = coords.detach().cpu().numpy()

        gt = {'states': np.vstack(x_prev),
              'values': np.vstack(true_v)}

        scio.savemat(os.path.join(save_root, f'train_data_t_{t:.2f}.mat'), gt)

    else:
        t_next = t - dt

        load_dir = os.path.join(logging_root, f'soccer_uncons_effort_square/t_{t_step - 1}/')

        val_model = icnn.SingleBVPNet(in_features=9, out_features=1, type=activation, mode='mlp',
                                      hidden_features=72, num_hidden_layers=5, dropout=0)
        val_model.to(device)
        model_path = os.path.join(load_dir, 'checkpoints_dir', 'model_final.pth')
        checkpoint = torch.load(model_path, map_location=device)
        try:
            val_model.load_state_dict(checkpoint['model'])
        except:
            val_model.load_state_dict(checkpoint)
        val_model.eval()


        x_next = point_dynamics(xy, u_high, d_high)

        X_next = torch.from_numpy(utils.make_pairs(x_next[:, :4], x_next[:, 4:8]))

        vs = []
        ps = np.linspace(0, 1, NUM_PS)

        for p_each in tqdm(ps):
            p_next = p_each * torch.ones_like(X_next[:, 0]).reshape(-1, 1)
            X_next_p = torch.cat((X_next, p_next), dim=1)
            # X_next_p = torch.cat((t_next * torch.ones((X_next_p.shape[0], 1)), X_next_p), dim=1)
            coords_in = {'coords': X_next_p.to(torch.float32)}
            # V_next = list(map(utils.final_cost, X_next[:, :2], X_next[:, 2:4],
            #                   [G for _ in range(X_next_p.shape[0])], p.detach().numpy()))
            V_next = val_model(coords_in)['model_out'].detach().numpy()

            V_next = V_next.reshape(-1, 9, 9) + dt * utils.inst_cost(u_high, d_high, R1, R2).reshape(-1, 9, 9)
            V_next = np.min(np.max(V_next, 2), 1)

            vs.append(V_next)

        true_v = utils.cav_vex(vs, type='vex', num_ps=NUM_PS).reshape(1, -1, 1)

        ps = torch.linspace(0, 1, 100)
        p = ps.repeat([len(xy), 1]).reshape(-1, 1)
        x = torch.vstack([xy[i].repeat([NUM_PS, 1]) for i in range(len(xy))])
        coords = torch.cat((x, p), dim=1)

        # time = torch.ones(x.shape[0], 1) * t
        # coords = torch.cat((time, coords), dim=1)

        x_prev = coords.detach().cpu().numpy()

        gt = {'states': np.vstack(x_prev),
              'values': np.vstack(true_v)}

        scio.savemat(os.path.join(save_root, f'train_data_t_{t:.2f}.mat'), gt)

