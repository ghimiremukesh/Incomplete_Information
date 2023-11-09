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
from odeintw import odeintw
import types

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

def dPdt(P, t, A, B, Q, R, S, ):
    n = A.shape[0]
    m = B.shape[1]

    if S is None:
        S = np.zeros((n, m))

    if isinstance(B, types.FunctionType):  # if B is time varying
        B_curr = B(t)
        B = B_curr

    return -(A.T @ P + P @ A - (P @ B + S) @ np.linalg.inv(R) @ (B.T @ P + S.T) + Q)


def dPhi(Phi, t, A):
    return np.dot(A, Phi)


def d(Phi, K, B, R, z):
    ds = np.zeros((len(Phi), 1))
    if isinstance(B, types.FunctionType):
        t_span = np.linspace(0, 1, 10)
        B_temp = np.array([B(i) for i in t_span])
    else:
        B_temp = np.array([B for _ in range(len(Phi))])

    B = B_temp
    for i in range(len(Phi)):
        # ds[i] = z.T @ Phi[i, :, :] @ K[i, :, :] @ B/R @ B.T @ K[i, :, :] @ Phi[i, :, :] @ z
        ds[i] = (z.T @ Phi[i, :, :].T @ K[i, :, :].T @ B[i] @ np.linalg.inv(R) @ B[i].T @ K[i, :, :] @ Phi[i, :, :] @ z)

    return ds


def value_hexner(x1, x2, p, t_step, Phi, K):
    """
    assuming R1 = R2 and A1 = A2, B1 = B2
    """
    z = np.array([[0], [1], [0], [0]])

    p1_val = p * (x1 - Phi[t_step, :, :] @ z).T @ K[t_step, :, :] @ (x1 - Phi[t_step, :, :] @ z) + \
             (1 - p) * (x1 + Phi[t_step, :, :] @ z).T @ K[t_step, :, :] @ (x1 + Phi[t_step, :, :] @ z)

    p2_val = p * (x2 - Phi[t_step, :, :] @ z).T @ K[t_step, :, :] @ (x2 - Phi[t_step, :, :] @ z) + \
             (1 - p) * (x2 + Phi[t_step, :, :] @ z).T @ K[t_step, :, :] @ (x2 + Phi[t_step, :, :] @ z)

    # value = np.sqrt(p1_val) - np.sqrt(p2_val)

    value = p1_val - p2_val

    return value

if __name__ == '__main__':
    model = 'particle'  # 'unicycle'
    num_points = 10
    num_players = 2
    num_states = 4  # x, y, vx, vy for each player


    # extra_points = 1000  # sample around unsafe region

    game = 'uncons'
    u_low = -2
    u_high = 2
    d_low = -1
    d_high = 1

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
    # xy_extra = torch.zeros(num_points, num_states * num_players).uniform_(-0.5, 0.5)
    # xy = torch.cat((xy, xy_extra), dim=0)

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






    # define system
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    # B1 = lambda t : np.array([[0], [t]])
    B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    # B1 = np.array([[0],[1]])
    Q = np.zeros((4, 4))
    R1 = 0.01 * np.eye(2, 2)
    # P1T = np.eye(2)
    PT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    tspan = np.linspace(0, 1, 11)
    tspan = np.flip(tspan)
    K1 = odeintw(dPdt, PT, tspan, args=(A, B, Q, R1, None,))

    K1 = np.flip(K1, axis=0)

    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    t_span = np.linspace(0, 1, 11)
    t_span = np.flip(t_span)
    PhiT = np.eye(4)

    Phi_sol = odeintw(dPhi, PhiT, t_span, args=(A,))
    Phi_sol = np.flip(Phi_sol, axis=0)

    z = np.array([[0], [1], [0], [0]])
    d1 = d(Phi_sol, K1, B, R1, z)
    B2 = B
    # B2 = lambda t : np.array([[0], [np.exp(-0.5*t)]])
    R2 = 0.01 * np.eye(2)
    K2 = odeintw(dPdt, PT, tspan, args=(A, B2, Q, R2, None,))
    K2 = np.flip(K2, axis=0)
    d2 = d(Phi_sol, K2, B2, R2, z)


    vals_hexner = []
    for i in range(len(x_prev)):
        value = value_hexner(x_prev[i, :4].reshape(-1, 1), x_prev[i, 4:8].reshape(-1, 1), x_prev[i, -1], 9, Phi_sol, K1)
        vals_hexner.append(value.item())


    check = vals_hexner == true_v.flatten()
