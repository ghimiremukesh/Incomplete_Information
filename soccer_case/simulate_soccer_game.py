# simulate the game
import numpy as np
import random

import utils
from solver_soccer import optimization
import icnn_pytorch_adaptive as icnn
import torch
import scipy.io as scio
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import matplotlib

# set nature distribution
u_max = 2
d_max = 1

us = list(product([-u_max, 0, u_max], repeat=2))
ds = list(product([-d_max, 0, d_max], repeat=2))
umap = {k: v for (k, v) in enumerate(us)}
dmap = {k: v for (k, v) in enumerate(ds)}


game = 'cons'  # for unconstrained game

p = 0.5  # probability of selecting goal 1

p1_type = random.choices([1, -1], [p, 1 - p])[0]
p1_type = -1
type_map = {1: 'Goal 1', -1: 'Goal 2'}

DT = 0.1
# initialize two models for first two timesteps, last value is known from the function


models = [icnn.SingleBVPNet(in_features=9, out_features=1, type='relu', mode='mlp', hidden_features=72,
                                    num_hidden_layers=5, dropout=0) for i in range(10)]


if game == 'cons':
    checkpoints = [f'logs/soccer_constrained/t_{i}/checkpoints_dir/model_final.pth'
                   for i in range(1, 11)]
else:
    checkpoints = [f'logs/soccer_uncons/t_{i}/checkpoints_dir/model_final.pth'
                   for i in range(1, 11)]

loaded_check = [torch.load(checkpoint, map_location=torch.device("cpu")) for checkpoint in checkpoints]

try:
    model_weights = [loaded['model'] for loaded in loaded_check]
except:
    model_weights = [loaded for loaded in loaded_check]

for i in range(len(models)):
    models[i].load_state_dict(model_weights[i])
    models[i].eval()



if __name__ == "__main__":

    states = []
    values = []
    print(f'Goal is: {type_map[p1_type]}')
    curr_x1 = np.array([-0.8, 0, 0, 0])  # x y pos vel for p1
    curr_x2 = np.array([-0.5, 0, 0, 0])
    print(f'Current position is : {curr_x1}, {curr_x2} and the belief is {p}\n')
    p_t = p
    # a_map = {'0': -1, '1': 0, '2': 1}

    t = 1  # backward time
    ts = np.arange(0, 1, DT)

    curr_pos = np.hstack((curr_x1, curr_x2, p_t))

    states.append(curr_pos)

    coords_in = torch.from_numpy(curr_pos).to(torch.float32)

    while t > DT:
        model_idx = int(10 * t) - 1
        curr_model = models[model_idx]
        next_model = models[model_idx - 1]
        feed_to_model = {'coords': coords_in}
        v_curr = curr_model(feed_to_model)['model_out'].detach().cpu().numpy()

        values.append(v_curr[0])
        (lam_j, p_1, p_2), u_1, u_2, d_1, d_2 = optimization(p_t, v_curr, curr_pos, t - DT, next_model, DT=DT, game=game)

        # if lam_j == 1:
        #     p_2 = 1 - p_1
        # else:
        #     p_2 = (p_t - lam_j * p_1) / (1 - lam_j)

        print(f'lamda_1 = {lam_j:.2f}, p_1 = {p_1:.2f},  p_2 = {p_2:.2f}\n')

        # action selection for first splitting point (lam_1)
        # calculate probability of each action
        if p1_type == -1:
            p_i = 1 - p_t
            p_1j = 1 - p_1
            p_2j = 1 - p_2
        else:
            p_i = p_t
            p_1j = p_1
            p_2j = p_2

        if lam_j == 1:
            a0_p = 1
            a1_p = 0
        else:
            a0_p = (lam_j * p_1j) / p_i
            a1_p = ((1 - lam_j) * p_2j) / p_i

        print(f'At t = {1-t:.2f}, P1 with type {type_map[p1_type]} has the following options: \n')
        print(f'P1 could take action {str(umap[u_1])} with probability {a0_p:.2f} and move belief to {p_1:.2f}')
        print(f'P1 could take action {str(umap[u_2])} with probability {a1_p:.2f} and move belief to {p_2:.2f}\n')

        dist = [a0_p, a1_p]
        a_idx = [0, 1]
        action_idx = random.choices(a_idx, dist)[0]
        # set to calculate strategy
        # action_idx = 0
        if action_idx == 0:
            action_1 = u_1
            action_2 = d_1  # assume p2 follows p1
            p_t = p_1
        else:
            action_1 = u_2
            action_2 = d_2  # assume p2 follows p1
            p_t = p_2

        # for simulation purpose select p2's action randomly
        # d_max = 1
        p2_ac = list(dmap.keys())
        p2_a = random.choices(p2_ac)[0]
        #
        # p2_action = 0 # set to always 0

        action_1 = umap[action_1]
        action_2 = dmap[action_2]
        # action_2 = dmap[p2_a]

          # p2 follows p1

        print(f'P1 chooses action: {action_1} and moves the belief to p_t = {p_t:.2f}')
        print(f'P2 chooses action: {action_2} (using minimax)')

        curr_x = utils.go_forward(curr_pos, action_1, action_2, DT)

        print(f'The current state is: {curr_x}\n')

        t = t - DT
        curr_pos = np.hstack((curr_x, p_t))
        states.append(curr_pos)


    # final time
    # data = {'states': states}
    # scio.savemat('plot_data.mat', data)

    states = np.vstack(states)

    # plt.plot(values)
    #
    dist_between = np.linalg.norm(states[:, :2] - states[:, 4:6], axis=1)
    plt.plot(np.linspace(0, 1, 11), dist_between, marker='o')
    plt.hlines(y=0.05, linestyles='--', xmin=0, xmax=1)

    x1 = states[:, 0]
    y1 = states[:, 1]
    x2 = states[:, 4]
    y2 = states[:, 5]

    p_t = states[:, -1]

    fig, axs = plt.subplots(2, 1)

    g1, g2 = utils.GOAL_1, utils.GOAL_2

    marker_angles_1 = np.arctan2(y1, x1)
    marker_angles_2 = np.arctan2(y2, x2)

    axs[0].set_title(f"Goal Selected: {type_map[p1_type]} ")
    if p1_type == -1:
        axs[0].scatter(g1[0], g1[1], marker='o', facecolor='none', edgecolor='magenta')
        axs[0].scatter(g2[0], g2[1], marker='o', facecolor='magenta', edgecolor='magenta')
    else:
        axs[0].scatter(g1[0], g1[1], marker='o', facecolor='magenta', edgecolor='magenta')
        axs[0].scatter(g2[0], g2[1], marker='o', facecolor='none', edgecolor='magenta')

    axs[0].annotate("1", (g1[0] + 0.01, g1[1]))
    axs[0].annotate("2", (g2[0] + 0.01, g2[1]))

    axs[0].scatter(x1[0], y1[0], marker='*', color='red')
    axs[0].scatter(x2[0], y2[0], marker='*', color='blue')
    axs[0].plot(x1, y1, color='red', label='A', marker='o', markersize=2)
    axs[0].plot(x2, y2, color='blue', label='D', marker='o', markersize=2)
    axs[0].set_xlim([-1, 1])
    axs[0].set_ylim([-1, 1])
    axs[0].legend()

    axs[1].plot(np.linspace(0, 1, 11), p_t)
    axs[1].set_xlabel('time (t)')
    axs[1].set_ylabel('belief (p_t)')

    # for i, angle in enumerate(marker_angles_1):
    #     axs[0].scatter(x1[i], y1[i], marker='>', transform=matplotlib.transforms.Affine2D().rotate(angle))

    plt.show()

    # for csv
    # states = np.vstack(states)
    # pos1 = states[:, 0]
    # time = np.flip(states[:, 0])
    # pos2 = states[:, 3]
    # vel1 = states[:, 2]
    # vel2 = states[:, 3]
    # belief = states[:, -1]
    #
    # d = {'time': time, 'pos1': pos1, 'vel1': vel1, 'pos2': pos2, 'vel2': vel2, 'belief': belief}
    # df = pd.DataFrame.from_dict(d)
    # df.to_csv('for_plots.csv')



