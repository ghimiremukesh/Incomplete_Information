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

    u_c = torch.tensor([-0.3, 0.3])
    d_c = torch.tensor([-0.1, 0.1])
    V_next = torch.zeros(2, 2)
    tau = 1e-3  # time step
    x_next = torch.clone(coords)

    for i in range(len(u_c)):
        for j in range(len(d_c)):
            # get the next state
            v = x_next[..., 2] + (u_c[i] - d_c[j]) * tau
            d = x_next[..., 1] + v * tau
            x_next[..., 1] = d
            x_next[..., 2] = v
            x_next[..., 0] = x_next[..., 0] + tau
            next_in = {'coords': x_next.to(device)}
            V_next[i, j] = model(next_in)['model_out'].squeeze()

    d_index = torch.argmax(V_next[:, :], dim=1)[1]
    u_index = torch.argmin(V_next[:, d_index])
    u = u_c[u_index]
    d = d_c[d_index]

    return u, d, y

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

        X_n = torch.cat((torch.tensor(t_nn, dtype=torch.float32, requires_grad=True),
                         torch.tensor(X_nn, dtype=torch.float32, requires_grad=True).T), dim=1)

        V = model({'coords': X_n.to(device)})['model_out']

        u_c = torch.tensor([-0.3, 0.3])
        d_c = torch.tensor([-0.1, 0.1])
        V_next = torch.zeros(2, 2)
        tau = 1e-3  # time step
        x_n = model({'coords': X_n.to(device)})['model_in']
        with torch.no_grad():
            x_n[-1] = p1
        x_next = [copy.deepcopy(x_n) for _ in range(len(u_c)) for _ in range(len(d_c))]

        count = 0
        for i in range(len(u_c)):
            for j in range(len(d_c)):
                # get the next state
                v = x_next[count][..., 2] + (u_c[i] - d_c[j]) * tau
                d = x_next[count][..., 1] + v * tau
                with torch.no_grad():
                    x_next[count][..., 1] = d
                    x_next[count][..., 2] = v
                    x_next[count][..., 0] = x_next[count][..., 0] - tau
                next_in = {'coords': x_next[count]}
                V_next[i, j] = model(next_in)['model_out'].squeeze()
                count += 1

        d_index = torch.argmax(V_next[:, :], dim=1)[1]
        u_index = torch.argmin(V_next[:, d_index])
        u = u_c[u_index]
        d = d_c[d_index]
        v_next_1 = V_next[u_index, d_index]

        x_n_2 = model({'coords': X_n.to(device)})['model_in']
        with torch.no_grad():
            x_n_2[-1] = p2
        x_next_2 = [copy.deepcopy(x_n_2) for _ in range(len(u_c)) for _ in range(len(d_c))]
        V_next_2 = torch.zeros(2, 2)
        count = 0
        for i in range(len(u_c)):
            for j in range(len(d_c)):
                # get the next state
                v = x_next_2[count][..., 2] + (u_c[i] - d_c[j]) * tau
                d = x_next_2[count][..., 1] + v * tau
                with torch.no_grad():
                    x_next_2[count][..., 1] = d
                    x_next_2[count][..., 2] = v
                    x_next_2[count][..., 0] = x_next_2[count][..., 0] - tau
                next_in = {'coords': x_next_2[count]}
                V_next_2[i, j] = model(next_in)['model_out'].squeeze()
                count += 1

        d_index = torch.argmax(V_next_2[:, :], dim=1)[1]
        u_index = torch.argmin(V_next_2[:, d_index])
        u = u_c[u_index]
        d = d_c[d_index]
        v_next_2 = V_next_2[u_index, d_index]

        # loss = V(t_k) - \sum lambda_i * V(t_k+1, p_i)
        loss = V - (lam_1 * v_next_1 + lam_2 * v_next_2)

        return abs(loss)

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

    if p_i == 0:
        p_i = p_i + 1e-2

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

    return U, D, P_t, (u_1, u_2, u_prob)


def dynamic(X_nn, dt, action):
    u1, u2 = action
    v = X_nn[1, :][0] + (u1 - u2) * dt
    d = X_nn[0, :][0] + v * dt

    return d, v

def helper(root, arr, ans):
    if not root:
        return

    arr.append(root.val.x[0])

    if root.left is None and root.right is None:
        # This will be only true when the node is leaf node
        # and hence we will update our ans array by inserting
        # array arr which have one unique path from root to leaf
        ans.append(arr.copy())
        del arr[-1]
        # after that we will return since we don't want to check after leaf node
        return

    # recursively going left and right until we find the leaf and updating the arr
    # and ans array simultaneously
    helper(root.left, arr, ans)
    helper(root.right, arr, ans)
    del arr[-1]


def Paths(root):
    # creating answer in which each element is a array
    # having one unique path from root to leaf
    ans = []
    # if root is null then there is no further action require so return
    if not root:
        return [[]]
    arr = []
    # arr is a array which will have one unique path from root to leaf
    # at a time.arr will be updated recursively
    helper(root, arr, ans)
    # after helper function call our ans array updated with paths so we will return ans array
    return ans


if __name__ == '__main__':

    logging_root = './logs'

    # Setting to plot
    ckpt_path = '../experiment_scripts/logs/training/checkpoints/model_final.pth'
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
        x0[:, 0] = 0  # put them in the center
        x0[:, 1] = 0

        # probability selections and calculations
        p_dist = np.random.rand()
        p_dist = 0.5  # for debugging
        p_dist = [p_dist, 1 - p_dist]
        types = [0, 1]
        type_i = np.random.choice(types, p=p_dist)  # nature selection from dist
        p_0 = p_dist[0]  # types_i = np.zeros(num_games) # random selection

        X0 = np.vstack((x0.T, p_0))

        N = 10
        Time = np.linspace(0, 1, num=N)
        dt = Time[1] - Time[0]
        Time = np.flip(Time)
        Time = Time[1:]
        time_t = np.array([1.])  # remove later
        for i in range(N - 1):
            time_t = np.hstack((time_t, np.repeat(Time[i], 2 ** (i + 1))))

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

        ini_state = State(x0.flatten().tolist(), None, p_0)  # initial state
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
            u, d, p_t, _ = optimization(X_nn, t_nn, dt, model, type_i)

            # expand left
            dx, dv = dynamic(X_nn, dt, (u[0], d[0]))
            dxdv = np.array([dx, dv]).flatten().tolist()
            current.left = Node(State(dxdv, u[0], p_t[0]))

            # expand right
            dx, dv = dynamic(X_nn, dt, (u[1], d[1]))
            dxdv = np.array([dx, dv]).flatten().tolist()
            current.right = Node(State(dxdv, u[1], p_t[1]))

        print(type_i)
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

        allstates = Paths(root)
        fig1, ax1 = plt.subplots()
        ax1.set_xlabel('Time-Steps')
        ax1.set_ylabel('Attacker\'s Position')
        for each in allstates:
            ax1.plot(each)

        plt.show()

        if save_data:
            save_path = f'relative_random_{i}.mat'
            scio.savemat(save_path, data)

    # plot the trajectory for all realizations
    # required vars: loop_index
    # states = np.zeros((30, N))
    # initial_state = root.val.x
    # states.append(initial_state[0])
    # current = root
    # # root
    # states[:, 0] = initial_state[0]  # initial state is same for all realizations
    # # branch 1
    # states[:15, 1] = root.left.val.x[0]  # set next state for first left child
    # states[15:, 1] = root.right.val.x[0]  # set next state for first right child
    #
    # # start adding for the left side of the tree
    # current = root.left
    # states[:7, 2] = current.left.val.x[0]  # set next state for second left child
    # states[7:15, 2] = current.right.val.x[0] # set next state for second right child

    #
    # while current.left is not None:












