import multiprocessing

import numpy as np
import torch
import icnn_pytorch_adaptive as icnn
import configargparse
import os
from soccer_case import utils
from itertools import product
from tqdm import tqdm
import scipy.io as scio

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
    vx = X[:, 2]
    vy = X[:, 3]

    xdot = vx
    ydot = vy

    us = product([u_low, 0, u_high], repeat=2)
    X_next = []

    for ux, uy in us:
        vxdot = ux
        vydot = uy
        x_new = x + xdot * dt + 0.5 * ux * dt ** 2
        y_new = y + ydot * dt + 0.5 * uy * dt ** 2
        vx_new = vx + vxdot * dt
        vy_new = vy + vydot * dt

        X_next.append(np.concatenate((x_new.reshape(-1, 1), y_new.reshape(-1, 1),
                                      vx_new.reshape(-1, 1), vy_new.reshape(-1, 1)), axis=1))

    return X_next

def point_dynamics_velocity(x, u_max, d_max, dt=0.1):
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


    x2 = x[:, 2].reshape(-1, 1)
    y2 = x[:, 3].reshape(-1, 1)




    x1dot = action_array_u[:, :, 0]
    y1dot = action_array_u[:, :, 1]

    x2dot = action_array_d[:, :, 0]
    y2dot = action_array_d[:, :, 1]

    x1_new = x1 + x1dot * dt
    y1_new = y1 + y1dot * dt

    x2_new = x2 + x2dot * dt
    y2_new = y2 + y2dot * dt

    X_new = np.hstack((x1_new.reshape(-1, 1), y1_new.reshape(-1, 1),
                       x2_new.reshape(-1, 1), y2_new.reshape(-1, 1)))

    return X_new
if __name__ == '__main__':
    model = 'particle'  # 'unicycle'
    num_points = 1000
    num_players = 2
    # num_states = 4  # x, y, vx, vy for each player

    num_states = 2  # x, y for velocity control

    extra_points = 1000  # sample around unsafe region

    u_low = -0.1
    u_high = 0.1
    d_low = -0.1
    d_high = 0.1

    g1 = utils.GOAL_1
    g2 = utils.GOAL_2

    G = [g1, g2]

    t = opt.time
    dt = 0.1

    ts = np.around(np.arange(dt, 1 + dt, dt), 2)
    t_step = int(np.where(ts == t)[0] + 1)

    NUM_PS = 100

    logging_root = 'logs/'
    save_root = f'soccer_velocity/'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    xy = torch.zeros(num_points, num_states * num_players).uniform_(-1, 1)

    xy_unsafe = torch.zeros(extra_points, num_states * num_players).uniform_(-0.02, 0.02)
    #
    xy = torch.cat((xy, xy_unsafe), dim=0)

    time = torch.ones(xy.shape[0], 1) * t

    if t == dt:
        t_next = t - dt
        x_next = np.vstack(point_dynamics_velocity(xy, u_high, d_high))

        X_next = torch.from_numpy(utils.make_pairs(x_next[:, :2], x_next[:, 2:]))

        vs = []
        ps = np.linspace(0, 1, NUM_PS)

        for p_each in tqdm(ps):
            p = p_each * torch.ones_like(X_next[:, 0]).reshape(-1, 1)
            X_next_p = torch.cat((X_next, p), dim=1)
            V_next = utils.final_cost(X_next[:, :2], X_next[:, 2:4], G, p.detach().numpy())
            # V_next = list(map(utils.final_cost, X_next[:, :2], X_next[:, 2:4],
            #                   [G for _ in range(X_next_p.shape[0])], p.detach().numpy()))

            V_next = V_next.reshape(-1, 9, 9)
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

    else:
        t_next = t - dt

        load_dir = os.path.join(logging_root, f'soccer/t_{t_step - 1}/')

        val_model = icnn.SingleBVPNet(in_features=5, out_features=1, type=activation, mode='mlp',
                                      hidden_features=128, num_hidden_layers=3, dropout=0)
        val_model.to(device)
        model_path = os.path.join(load_dir, 'checkpoints_dir', 'model_final.pth')
        checkpoint = torch.load(model_path, map_location=device)
        try:
            val_model.load_state_dict(checkpoint['model'])
        except:
            val_model.load_state_dict(checkpoint)
        val_model.eval()

        x_next_1 = np.vstack(point_dynamics(xy[:, :num_states], u_low, u_high))
        x_next_2 = np.vstack(point_dynamics(xy[:, num_states:], d_low, d_high))

        X_next = torch.from_numpy(utils.make_pairs_vel(x_next_1, x_next_2))

        vs = []
        ps = np.linspace(0, 1, NUM_PS)

        for p_each in tqdm(ps):
            p = p_each * torch.ones_like(X_next[:, 0]).reshape(-1, 1)
            X_next_p = torch.cat((X_next, p), dim=1)
            # X_next_p = torch.cat((t_next * torch.ones((X_next_p.shape[0], 1)), X_next_p), dim=1)
            coords_in = {'coords': X_next_p.to(torch.float32)}
            # V_next = list(map(utils.final_cost, X_next[:, :2], X_next[:, 2:4],
            #                   [G for _ in range(X_next_p.shape[0])], p.detach().numpy()))
            V_next = val_model(coords_in)['model_out'].detach().numpy()

            V_next = V_next.reshape(-1, 3, 3)
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

