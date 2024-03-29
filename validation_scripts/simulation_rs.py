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
    opt_x = min(reduced, key=objective_v)  # find 3-uple corresponding to min objective func.

    p = X_nn[-1, :]

    if type == 1:  # p_i corresponds to which type you are
        p_i = 1 - p
    else:
        p_i = p

    if p == 0:
        p = p + 1e-2

    lamb = opt_x[0]
    p1 = opt_x[1]
    p2 = opt_x[2]
    u_prob = np.array([lamb * p1 / p_i, (1 - lamb) * p2 / p_i])

    X_u1 = copy.deepcopy(X_nn)
    X_u1[-1] = p1
    u_1, d_1, _ = value_action(X_u1, t_nn, model)

    X_u2 = copy.deepcopy(X_nn)
    X_u2[-1] = p2
    u_2, d_2, _ = value_action(X_u2, t_nn, model)

    if u_1 == u_2:
        P_t = p
        U, D, _ = value_action(X_nn, t_nn, model)
    else:
        # Pick u1 from the distribution u_prob
        u_idx = np.array([0, 1])  # action candidates
        U_idx = random.choices(u_idx, list(u_prob.flatten()))[0]
        index = [i for i in range(len(u_idx)) if u_idx[i] == U_idx][0]

        P_t = p1 if index == 0 else p2  # pick p_j corresponding to u_j
        U = u_1 if index == 0 else u_2
        D = d_1 if index == 0 else d_2

    return U, D, P_t, (u_1, u_2, u_prob)

if __name__ == '__main__':

    logging_root = './logs'

    # Setting to plot
    ckpt_path = '../logs/random_p/checkpoints/model_final.pth'
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
        x0[:, 0] = 0.1 # put them in the center
        x0[:, 1] = 0

        # probability selections and calculations
        p_dist = np.random.rand()
        p_dist = 0.047  # for debugging
        p_dist = [p_dist, 1 - p_dist]
        types = [0, 1]
        type_i = np.random.choice(types, p=p_dist)  # nature selection from dist
        p_0 = p_dist[0]  # types_i = np.zeros(num_games) # random selection


        X0 = np.vstack((x0.T, p_0))

        N = 50
        Time = np.linspace(0, 1, num=N)
        dt = Time[1] - Time[0]
        Time = np.flip(Time)

        d = np.zeros((N,))
        v = np.zeros((N,))
        u1 = np.zeros((N,))
        u2 = np.zeros((N,))
        p = np.zeros((N,))
        V = np.zeros((N,))

        d[0] = X0[0, :]
        v[0] = X0[1, :]
        p[0] = X0[2, :]
        splitting_points = []

        start_time = time.time()

        for j in range(1, Time.shape[0] + 1):
            X_nn = np.array([[d[j - 1]], [v[j - 1]], [p[j - 1]]])
            t_nn = np.array([[Time[j - 1]]])
            _, _, V[j - 1] = value_action(X_nn, t_nn, model)

            u1[j - 1], u2[j - 1], p_t, sp = optimization(X_nn, t_nn, dt, model, type_i)
            if j == Time.shape[0]:
                break
            else:
                d[j], v[j] = dynamic(X_nn, dt, (u1[j - 1], u2[j - 1]))
                p[j] = p_t
                splitting_points.append(sp)
            print(j)

        print()
        time_spend = time.time() - start_time
        print('Total solution time: %1.1f' % (time_spend), 'sec')
        print()

        data = {'d': d,
                'v': v,
                'u1': u1,
                'u2': u2,
                'p': p,
                'V': V,
                't': np.flip(Time),
                'type': type_i,
                'p_0': p_0}

        save_data = 1  # input('Save data? Enter 0 for no, 1 for yes:')

        if save_data:
            save_path = f'relative_random_{i}.mat'
            scio.savemat(save_path, data)

    # For testing remove later.
    # x_plot = torch.zeros(10, 3).uniform_(-1, 1)
    # x_plot[:, 0] = 0.5 * torch.ones_like(x_plot[:, 0])
    # time = torch.zeros(10, 1)# final time
    # coords = torch.cat((time, x_plot), dim=1)
    # coords_in = {'coords': coords}
    # vals = model(coords_in)['model_out'].detach().cpu().numpy().squeeze()
    #
    # p = x_plot[:, -1].detach().numpy().squeeze()
    #
    # plt.plot(p, vals, 'r-')
    # plt.xlabel('p')
    # plt.ylabel('V')
    # plt.title('Value at final time for a fixed state')
    # plt.show()
