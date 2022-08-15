# Enable import from parent package
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules, diff_operators, modules_picnn
import time
import torch
import numpy as np
import scipy.io as scio
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import copy
import random
from itertools import product

from utils.StateObject import State
from utils.BinaryTreeMod import Node


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def value_action(X_nn, t_nn, model):
    d = X_nn[0]
    v = X_nn[1]
    p = X_nn[2]

    X = np.vstack((d, v, p))
    X = torch.tensor(X, dtype=torch.float32, requires_grad=True).T
    t = torch.tensor(t_nn, dtype=torch.float32, requires_grad=True)
    coords = torch.cat((t, X), dim=1)

    model_in = {'coords': coords.to(device)}
    model_output = model(model_in)

    x = model_output['model_in']
    y = model_output['model_out']

    jac, _ = diff_operators.jacobian(y, x)

    # partial gradient of V w.r.t. time and state
    dvdt = jac[..., 0, 0]
    dvdx = jac[..., 0, 1:]

    # costates
    lam_d = dvdx[:, :, :1].squeeze()
    lam_v = dvdx[:, :, 1:2].squeeze()

    u_c = torch.tensor([-0.3, 0.3]).to(device)
    d_c = torch.tensor([-0.1, 0.1]).to(device)
    v = torch.tensor(np.array([v])).squeeze().to(device)
    H = torch.zeros(2, 2)

    for i in range(len(u_c)):
        for j in range(len(d_c)):
            H[i, j] = lam_d * v + lam_v * (u_c[i] - d_c[j])

    d_index = torch.argmax(H[:, :], dim=1)[1]
    u_index = torch.argmin(H[:, d_index])  # minmax
    u = u_c[u_index]
    d = d_c[d_index]

    u = u.detach().cpu().numpy()
    d = d.detach().cpu().numpy()
    y = y.detach().cpu().numpy().squeeze()

    return u, d, y


def hji_compute(X_nn, t_nn, model):
    d = X_nn[0]
    v = X_nn[1]
    p = X_nn[2]

    X = np.vstack((d, v, p))
    X = torch.tensor(X, dtype=torch.float32, requires_grad=True).T
    t = torch.tensor(t_nn, dtype=torch.float32, requires_grad=True)
    coords = torch.cat((t, X), dim=1)

    model_in = {'coords': coords.to(device)}
    model_output = model(model_in)

    x = model_output['model_in']
    y = model_output['model_out']

    jac, _ = diff_operators.jacobian(y, x)

    # partial gradient of V w.r.t. time and state
    dvdt = jac[..., 0, 0]
    dvdx = jac[..., 0, 1:]

    # unnormalize the costate for agent 1
    lam_d = dvdx[:, :, :1].squeeze()
    lam_v = dvdx[:, :, -1:].squeeze()

    u_c = torch.tensor([-0.3, 0.3]).to(device)
    d_c = torch.tensor([-0.1, 0.1]).to(device)
    v = torch.tensor(np.array([v])).squeeze().to(device)
    H = torch.zeros(2, 2)

    for i in range(len(u_c)):
        for j in range(len(d_c)):
            H[i, j] = lam_d * v + lam_v * (u_c[i] - d_c[j])

    d_index = torch.argmax(H[:, :], dim=1)[1]
    u_index = torch.argmin(H[:, d_index])  # minmax
    u = u_c[u_index]
    d = d_c[d_index]

    ham = lam_d * v + lam_v * (u - d)
    hji = -dvdt + ham
    hji = hji.detach().cpu().numpy().squeeze()
    y = y.detach().cpu().numpy().squeeze()

    return hji, y


def dynamic(X_nn, dt, action):
    u1, u2 = action
    v = X_nn[1, :] + (u1 - u2) * dt
    d = X_nn[0, :] + v * dt

    return d, v


def optimization(X_nn, t_nn, dt, model, type):
    # \sum lambda_j * p_j = p
    def constraint(var):
        constrain = var[0] * var[1] + (1 - var[0]) * var[2] - X_nn[-1]
        return abs(constrain) <= 5e-3

    def objective_v(var):
        lam_1 = var[0]
        lam_2 = 1 - var[0]
        p1 = var[1]
        p2 = var[2]

        _, _, V = value_action(X_nn, t_nn, model)

        # 1. obtain action for splitting point 1 and 2
        # 2. compute value for splitting point 1 and 2
        # splitting point 1
        X_nn1 = copy.deepcopy(X_nn)
        X_nn1[-1] = p1
        u1_s, u2_s, _ = value_action(X_nn1, t_nn, model)
        d1, v1 = dynamic(X_nn1, dt, (u1_s, u2_s))

        X_nn1 = np.vstack((d1, v1, p1))
        t_nn1 = t_nn - dt
        _, _, V1 = value_action(X_nn1, t_nn1, model)

        # splitting point 2
        X_nn2 = copy.deepcopy(X_nn)
        X_nn2[-1] = p2
        u1_s, u2_s, _ = value_action(X_nn2, t_nn, model)
        d2, v2 = dynamic(X_nn1, dt, (u1_s, u2_s))

        X_nn2 = np.vstack((d2, v2, p2))
        t_nn2 = t_nn - dt
        _, _, V2 = value_action(X_nn2, t_nn2, model)

        # loss = V(t_k) - \sum lambda_i * V(t_k+1, p_i)
        loss = V - (lam_1 * V1 + lam_2 * V2)

        return loss

    def objective(var):
        lam_1 = var[0]
        lam_2 = 1 - var[0]
        p1 = var[1]
        p2 = var[2]

        _, _, V = value_action(X_nn, t_nn, model)

        # 1. obtain action for splitting point 1 and 2
        # 2. compute value for splitting point 1 and 2

        # splitting point 1
        X_nn1 = copy.deepcopy(X_nn)
        X_nn1[-1] = p1
        hji_1, V1 = hji_compute(X_nn1, t_nn, model)

        # splitting point 2
        X_nn2 = copy.deepcopy(X_nn)
        X_nn2[-1] = p2
        hji_2, V2 = hji_compute(X_nn2, t_nn, model)

        # loss = V(t_k) - \sum lambda_i * V(t_k+1, p_i)
        loss = V - (lam_1 * (V1 + hji_1 * dt) + lam_2 * (V2 + hji_2 * dt))

        return loss

    search_space = np.linspace(0, 1, num=11)
    # 1-D grid search for lambda, p1, p2
    grid = product(search_space, repeat=3)  # make a grid
    reduced = filter(constraint, grid)  # apply filter to reduce the space
    opt_x = min(reduced, key=objective)  # find 3-uple corresponding to min objective func.

    p = X_nn[-1, :]

    if type == 1:  # p_i corresponds to which type you are
        p_i = 1 - p
    else:
        p_i = p

    if p_i == 0:
        p_i = p_i + 1e-2

    lamb = opt_x[0]
    p1 = opt_x[1]
    p2 = opt_x[2]

    if type == 1:
        u_prob = np.array([lamb * (1 - p1) / p_i, (1 - lamb) * (1 - p2) / p_i])
    else:
        u_prob = np.array([lamb * p1 / p_i, (1 - lamb) * p2 / p_i])

    X_u1 = copy.deepcopy(X_nn)
    X_u1[-1] = p1
    u_1, d_1, _ = value_action(X_u1, t_nn, model)

    X_u2 = copy.deepcopy(X_nn)
    X_u2[-1] = p2
    u_2, d_2, _ = value_action(X_u2, t_nn, model)

    if u_1 == u_2:
        u, d, _ = value_action(X_nn, t_nn, model)
        U = [u, u]
        D = [d, d]
        P_t = [p, p]
    else:
        # # Pick u1 from the distribution u_prob
        # u_idx = np.array([0, 1])  # action candidates
        # U_idx = random.choices(u_idx, list(u_prob.flatten()))[0]
        # index = [i for i in range(len(u_idx)) if u_idx[i] == U_idx][0]
        #
        # P_t = p1 if index == 0 else p2  # pick p_j corresponding to u_j
        # U = u_1 if index == 0 else u_2
        # D = d_1 if index == 0 else d_2

        U = [u_1, u_2]
        D = [d_1, d_2]
        P_t = [p1, p2]

    return U, D, P_t


if __name__ == '__main__':

    logging_root = './logs'

    # Setting to plot
    ckpt_path = '../logs/random_p_test_revisit/checkpoints/model_final.pth'
    # ckpt_path = '../experiment_scripts/logs/4d_picnn_min_hji/checkpoints/model_final.pth'
    activation = 'tanh'

    # Initialize and load the model
    model = modules_picnn.SingleBVPNet(in_features=4, out_features=1, type=activation, mode='mlp',
                                       final_layer_factor=1., hidden_features=32, num_hidden_layers=3)
    model.to(device)
    if device == torch.device("cpu"):
        checkpoint = torch.load(ckpt_path, map_location=device)
    else:
        checkpoint = torch.load(ckpt_path)

    try:
        model_weights = checkpoint['model']
    except:
        model_weights = checkpoint
    model.load_state_dict(model_weights)
    model.eval()

    num_games = 1
    # p_dist = [0.6, 0.4]
    # types = [0, 1]
    # types_i = np.random.choice(types, size=num_games, p=p_dist) # nature selection from dist
    # p_0 = p_dist[0]# types_i = np.zeros(num_games) # random selection
    for i in range(num_games):
        num_physical = 2
        x0 = torch.zeros(1, num_physical).uniform_(-1, 1)
        x0[:, 0] = 0.1  # put them in the center
        x0[:, 1] = 0

        # probability selections and calculations
        p_dist = np.random.rand()
        p_dist = 0.8  # for debugging
        p_dist = [p_dist, 1 - p_dist]
        types = [0, 1]
        type_i = np.random.choice(types, p=p_dist)  # nature selection from dist
        p_0 = p_dist[0]  # types_i = np.zeros(num_games) # random selection


        X0 = np.vstack((x0.T, p_0))

        N = 5
        Time = np.linspace(0, 1, num=N)
        dt = Time[1] - Time[0]
        Time = np.flip(Time)
        Time = Time[1:]
        time_t = np.array([1.])  # remove later
        for i in range(N - 1):
            time_t = np.hstack((time_t, np.repeat(Time[i], 2**(i+1))))

        time_t = time_t.tolist()
        # time_t.pop(0) # remove first element

        d_list = list()
        v_list = list()
        u1_list = list()
        u2_list = list()
        p_list = list()
        V_list = list()

        d_list.append([])
        v_list.append([])
        p_list.append([])

        d_list[0].append(X0[0, :])
        v_list[0].append(X0[1, :])
        p_list[0].append(X0[2, :])

        start_time = time.time()

        ini_state = State(x0.flatten().tolist(), None, p_0) # initial state
        root = Node(ini_state)

        loop_index = 2 ** (N - 1) - 1
        for i in range(loop_index):
            current = root[i]  # get current node
            # check if children exist. possible number of children always 2.
            while current.left is not None:
                current = current.left

            dv = np.array(current.val.get_state()).reshape(-1, 1)
            p = np.array(current.val.get_belief()).reshape(-1, 1)
            X_nn = np.vstack((dv, p))
            t_nn = np.array([[time_t[i]]])
            _, _, V = value_action(X_nn, t_nn, model)
            u, d, p_t = optimization(X_nn, t_nn, dt, model, type_i)

            # expand left
            dx, dv = dynamic(X_nn, dt, (u[0], d[0]))
            dxdv = np.array([dx, dv]).flatten().tolist()
            current.left = Node(State(dxdv, u[0], p_t[0]))

            # expand right
            dx, dv = dynamic(X_nn, dt, (u[1], d[1]))
            dxdv = np.array([dx, dv]).flatten().tolist()
            current.right = Node(State(dxdv, u[1], p_t[1]))

        print(root)


        time_spend = time.time() - start_time
        print('Total solution time: %1.1f' % (time_spend), 'sec')


        data = {'d': d_list,
                'v': v_list,
                'u1': u1_list,
                'u2': u2_list,
                'p': p_list,
                'V': V_list,
                't': np.flip(Time),
                'type': type_i,
                'p_0': p_0}

        save_data = 0  # input('Save data? Enter 0 for no, 1 for yes:')

        if save_data:
            save_path = f'relative_random_{i}.mat'
            scio.savemat(save_path, data)
