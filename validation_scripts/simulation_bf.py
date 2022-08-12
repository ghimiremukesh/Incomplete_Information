# Enable import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules, diff_operators, modules_picnn, modules_ficnn
import time
import torch
import numpy as np
import scipy.io as scio
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import copy
import random
from itertools import product
import multiprocessing

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def value_action(X_nn, t_nn, model):
    d1 = X_nn[0]
    v1 = X_nn[1]
    d2 = X_nn[2]
    v2 = X_nn[3]
    p = X_nn[4]

    X = np.vstack((d1, v1, d2, v2, p))
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
    lam_da = dvdx[:, :, :1].squeeze()
    lam_va = dvdx[:, :, 1:2].squeeze()
    lam_dd = dvdx[:, :, 2:3].squeeze()
    lam_vd = dvdx[:, :, 3:4].squeeze()

    u_c = torch.tensor([-0.3, 0.3]).cuda()
    d_c = torch.tensor([-0.1, 0.1]).cuda()
    v1 = torch.tensor([v1]).squeeze().cuda()
    v2 = torch.tensor([v2]).squeeze().cuda()
    H = torch.zeros(2, 2)

    for i in range(len(u_c)):
        for j in range(len(d_c)):
            H[i, j] = lam_da * v1 + lam_va * u_c[i] + lam_dd * v2 + lam_vd * d_c[j]

    d_index = torch.argmax(H[:, :], dim=1)[1]
    u_index = torch.argmin(H[:, d_index])  # minmax
    u = u_c[u_index]
    d = d_c[d_index]

    u = u.detach().cpu().numpy()
    d = d.detach().cpu().numpy()
    y = y.detach().cpu().numpy().squeeze()

    return u, d, y

def hji_compute(X_nn, t_nn, model):
    d1 = X_nn[0]
    v1 = X_nn[1]
    d2 = X_nn[2]
    v2 = X_nn[3]
    p = X_nn[4]

    X = np.vstack((d1, v1, d2, v2, p))
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
    lam_da = dvdx[:, :, :1].squeeze()
    lam_va = dvdx[:, :, 1:2].squeeze()
    lam_dd = dvdx[:, :, 2:3].squeeze()
    lam_vd = dvdx[:, :, 3:4].squeeze()

    u_c = torch.tensor([-0.3, 0.3]).cuda()
    d_c = torch.tensor([-0.1, 0.1]).cuda()
    v1 = torch.tensor([v1]).squeeze().cuda()
    v2 = torch.tensor([v2]).squeeze().cuda()
    H = torch.zeros(2, 2)

    for i in range(len(u_c)):
        for j in range(len(d_c)):
            H[i, j] = lam_da * v1 + lam_va * u_c[i] + lam_dd * v2 + lam_vd * d_c[j]

    d_index = torch.argmax(H[:, :], dim=1)[1]
    u_index = torch.argmin(H[:, d_index])  # minmax
    u = u_c[u_index]
    d = d_c[d_index]

    ham = lam_da * v1 + lam_va * u + lam_dd * v2 + lam_vd * d
    # hji = dvdt - ham
    hji = -dvdt + ham

    hji = hji.detach().cpu().numpy().squeeze()
    y = y.detach().cpu().numpy().squeeze()

    return hji, y

def dynamic(X_nn, dt, action):
    u1, u2 = action
    v1 = X_nn[1, :] + u1 * dt
    v2 = X_nn[3, :] + u2 * dt
    d1 = X_nn[0, :] + v1 * dt
    d2 = X_nn[2, :] + v2 * dt

    return d1, v1, d2, v2

def optimization(X_nn, t_nn, dt, model, type):
    # def objective(var, X_nn, t_nn, dt, model):
    def objective(var):
        lam_1 = var[0]
        lam_2 = 1 - var[0]
        p1 = var[1]
        p2 = var[2]

        _, _, V = value_action(X_nn, t_nn, model)

        # 1. obtain action for splitting point 1 and 2
        # 2. compute value for splitting point 1 and 2
        # Taylor expansion for V(t+dt, x(t+dt), p)
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

        # firstly consider u*, d* from V(t, x(t), p), then do forward dynamic and compute V(t+dt, x(t+dt), p)
        # # splitting point 1
        # X_nn1 = copy.deepcopy(X_nn)
        # X_nn1[-1] = p1
        # u1_s, u2_s, _ = value_action(X_nn1, t_nn, model)
        # d1, v1, d2, v2 = dynamic(X_nn1, dt, (u1_s, u2_s))
        #
        # X_nn1 = np.vstack((d1, v1, d2, v2, p1))
        # t_nn1 = t_nn - dt
        # _, _, V1 = value_action(X_nn1, t_nn1, model)
        #
        # # splitting point 2
        # X_nn2 = copy.deepcopy(X_nn)
        # X_nn2[-1] = p2
        # u1_s, u2_s, _ = value_action(X_nn2, t_nn, model)
        # d1, v1, d2, v2 = dynamic(X_nn1, dt, (u1_s, u2_s))
        #
        # X_nn2 = np.vstack((d1, v1, d2, v2, p2))
        # t_nn2 = t_nn - dt
        # _, _, V2 = value_action(X_nn2, t_nn2, model)
        #
        # # loss = V(t_k) - \sum lambda_i * V(t_k+1, p_i)
        # loss = V - (lam_1 * V1 + lam_2 * V2)

        return loss

    # \sum lambda_j * p_j = p
    # def constraint(var, X_nn):
    def constraint(var):
        constrain = var[0] * var[1] + (1 - var[0]) * var[2] - X_nn[-1]
        return abs(constrain) <= 5e-3
        # return abs(constrain)

    Lam = np.linspace(0, 1, num=11)
    P1 = np.linspace(0, 1, num=11)
    P2 = np.linspace(0, 1, num=11)

    opt_sol = {'sol': [],
               'opt_x': []}

    # 1-D grid search for lambda, p1, p2
    grid = product(Lam, P1, P2)  # make a grid
    reduced = filter(constraint, grid)  # apply filter to reduce the space
    opt_x = min(reduced, key=objective)  # find 3-uple corresponding to min objective func.

    # for (lam, p1, p2) in product(Lam, P1, P2):
    #     var = np.array([lam, p1, p2])
    #     constrain = constraint(var, X_nn)
    #
    #     if constrain <= 5e-3:
    #         sol = objective(var, X_nn, t_nn, dt, model)
    #         opt_sol['sol'].append(np.array([sol]))
    #         opt_sol['opt_x'].append(np.array([lam, p1, p2]))

    # index = np.argmin(opt_sol['sol'])
    # opt_x = opt_sol['opt_x'][index]

    p = X_nn[-1, :]

    lamb = opt_x[0]
    p1 = opt_x[1]
    p2 = opt_x[2]

    if type == 1:  # p_i corresponds to which type you are
        p_i = 1 - p
    else:
        p_i = p

    if p_i == 0:
        p_i = p_i + 1e-2

    if type == 1:
        u_prob = np.array([lamb * (1 - p1) / p_i, (1 - lamb) * (1 - p2) / p_i])
    else:
        u_prob = np.array([lamb * p1 / p_i, (1 - lamb) * p2 / p_i])

    X_u1 = copy.deepcopy(X_nn)
    X_u1[-1] = p1
    u_1, d_1, V_1 = value_action(X_u1, t_nn, model)

    X_u2 = copy.deepcopy(X_nn)
    X_u2[-1] = p2
    u_2, d_2, V_2 = value_action(X_u2, t_nn, model)

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

    return U, D, P_t

if __name__ == '__main__':

    logging_root = './logs'

    # Setting to plot
    # ckpt_path = '../experiment_scripts/logs/min hji/picnn_arch_test_0.8/checkpoints/model_final.pth'
    ckpt_path = '../experiment_scripts/logs/picnn_arch_test1/checkpoints/model_final.pth'
    activation = 'tanh'

    # Initialize and load the model
    model = modules_picnn.SingleBVPNet(in_features=6, out_features=1, type=activation, mode='mlp',
                                 final_layer_factor=1., hidden_features=32, num_hidden_layers=3)
    model.cuda()
    checkpoint = torch.load(ckpt_path)
    try:
        model_weights = checkpoint['model']
    except:
        model_weights = checkpoint
    model.load_state_dict(model_weights)
    model.eval()

    num_games = 10
    for i in range(num_games):
        num_physical = 4
        x0 = torch.zeros(1, num_physical).uniform_(-1, 1)
        x0[:, 0] = 0  # put them in the center
        x0[:, 2] = 0
        x0[:, 1] = 0
        x0[:, 3] = 0

        # probability selections and calculations
        p_dist = np.random.rand()
        # p_dist = 0.047  # for debugging
        p_dist = [p_dist, 1 - p_dist]
        types = [0, 1]
        type_i = np.random.choice(types, p=p_dist)  # nature selection from dist
        p_0 = p_dist[0]  # types_i = np.zeros(num_games) # random selection
        X0 = np.vstack((x0.T, p_0))

        N = 50
        Time = np.linspace(0, 1, num=N)
        dt = Time[1] - Time[0]
        Time = np.flip(Time)

        d1 = np.zeros((N,))
        v1 = np.zeros((N,))
        u1 = np.zeros((N,))
        d2 = np.zeros((N,))
        v2 = np.zeros((N,))
        u2 = np.zeros((N,))
        p = np.zeros((N,))
        V = np.zeros((N,))

        d1[0] = X0[0, :]
        v1[0] = X0[1, :]
        d2[0] = X0[2, :]
        v2[0] = X0[3, :]
        p[0] = X0[4, :]

        start_time = time.time()

        for j in range(1, Time.shape[0] + 1):
            X_nn = np.array([[d1[j - 1]], [v1[j - 1]], [d2[j - 1]], [v2[j - 1]], [p[j - 1]]])
            t_nn = np.array([[Time[j - 1]]])
            _, _, V[j - 1] = value_action(X_nn, t_nn, model)

            u1[j - 1], u2[j - 1], p_t = optimization(X_nn, t_nn, dt, model, type_i)
            if j == Time.shape[0]:
                break
            else:
                d1[j], v1[j], d2[j], v2[j] = dynamic(X_nn, dt, (u1[j - 1], u2[j - 1]))
                p[j] = p_t
            print(j)

        print()
        time_spend = time.time() - start_time
        print('Total solution time: %1.1f' % (time_spend), 'sec')
        print()

        data = {'d1': d1,
                'd2': d2,
                'v1': v1,
                'v2': v2,
                'u1': u1,
                'u2': u2,
                'p': p,
                'V': V,
                'type': type_i,
                't': np.flip(Time)}

        save_data = 1 #input('Save data? Enter 0 for no, 1 for yes:')

        if save_data:
            save_path = 'hji_soccer_case' + str(i) + '.mat'
            scio.savemat(save_path, data)